//! VeriBin's canonicalization, ported from the pre-clarirs claripy patch.
//!
//! VeriBin decides whether two decompiled functions agree by comparing symbolic
//! expressions. Before handing a pair to z3 it asks a cheaper question: are
//! these two ASTs the same *shape*, ignoring variable names, the order of
//! commutative operands, and the direction of a reversible comparison?
//! [`canonical_hash`] answers that, and [`canonicalize`] produces an actual AST
//! with those differences normalized away so two of them can be compared as
//! strings.
//!
//! Only the policy lives here. The traversal is clarirs' [`walk`] (iterative,
//! and it caches per node so a shared subexpression is visited once), the
//! rebuild is clarirs' [`reconstruct_node`], and the memo is clarirs'
//! [`GenericCache`]. What clarirs has no form for is the VeriBin policy itself:
//! its op classification, its variable-path hash, and its notion of a canonical
//! form. clarirs' own `algorithms::canonicalize` is a different algorithm --
//! it renames variables lexicographically and returns a 3-tuple angr depends
//! on, with no hash, no operand sorting and no comparison flipping.
//!
//! Faithfulness notes -- two quirks of the original are preserved on purpose,
//! because changing them changes which functions VeriBin calls equivalent:
//!
//! * Renaming numbers leaves per *occurrence*, not per distinct variable, so
//!   `x + x` normalizes to `var_0 + var_1` -- the same as `x + y`.
//! * The operator reaches the hash only through variable paths, so a subtree
//!   with no variables is hashed op-blind: `1 + 2` and `1 * 2` hash equal.
//!
//! `canonical_hash` values are only ever in-process dict keys (never persisted,
//! never compared across runs), so they need not match the Python
//! implementation's numbers. Only the equivalence classes must, and those do.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, LazyLock};

use clarirs_core::algorithms::reconstruct::reconstruct_node;
use clarirs_core::algorithms::walk;
use clarirs_core::cache::GenericCache;

use crate::claripy::prelude::*;

// --------------------------------------------------------------- op classes
//
// Classification is by op *string*, not by `AstOp` variant, because that is
// what the original keyed on and the two do not coincide: `Eq` on floats is
// `fpEQ`, which is not in the commutative set, while `Eq` on bitvectors is
// `__eq__`, which is.

/// Ops that introduce a symbol. Their name is dropped when hashing, which is
/// what makes two differently-named variables of the same sort compare equal.
fn is_symbol_creation(op: &str) -> bool {
    matches!(op, "BVS" | "BoolS" | "FPS" | "StringS")
}

/// Ops that introduce a literal.
fn is_value_creation(op: &str) -> bool {
    matches!(op, "BVV" | "BoolV" | "FPV" | "StringV")
}

/// Ops whose operand order carries no meaning. `__eq__`/`__ne__` are VeriBin
/// additions: they are commutative for canonicalization even though clarirs'
/// own simplifier is not told so (it is shared with angr's symbolic execution
/// and must not gain new rewrites).
fn is_commutative(op: &str) -> bool {
    matches!(
        op,
        "__and__"
            | "__or__"
            | "__xor__"
            | "__add__"
            | "__mul__"
            | "And"
            | "Or"
            | "Xor"
            | "__eq__"
            | "__ne__"
    )
}

/// Comparisons that mean the same thing with their operands swapped. Only the
/// "less" direction is listed, so canonicalization always lands on the "greater"
/// form. claripy's table also carried `__lt__`, `__radd__` and friends; clarirs
/// never emits those op strings, so they are dropped rather than ported dead.
fn reversed_op(op: &str) -> Option<&'static str> {
    match op {
        "ULT" => Some("UGT"),
        "ULE" => Some("UGE"),
        "SLT" => Some("SGT"),
        "SLE" => Some("SGE"),
        _ => None,
    }
}

/// The reversed node itself, for [`canonicalize`]. Mirrors [`reversed_op`].
fn reverse(ast: &AstRef<'static>) -> Result<Option<AstRef<'static>>, ClaripyError> {
    let op = match ast.op() {
        AstOp::ULT(a, b) => AstOp::UGT(b.clone(), a.clone()),
        AstOp::ULE(a, b) => AstOp::UGE(b.clone(), a.clone()),
        AstOp::SLT(a, b) => AstOp::SGT(b.clone(), a.clone()),
        AstOp::SLE(a, b) => AstOp::SGE(b.clone(), a.clone()),
        _ => return Ok(None),
    };
    Ok(Some(GLOBAL_CONTEXT.make_ast(op)?))
}

// ------------------------------------------------------------- hash helpers

fn hash_of<T: std::hash::Hash>(value: &T) -> u64 {
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    std::hash::Hash::hash(value, &mut hasher);
    hasher.finish()
}

/// Hash of a value that is not itself hashable but is uniquely described by its
/// debug form (float sorts, rounding modes, float literals).
fn hash_debug<T: std::fmt::Debug>(value: &T) -> u64 {
    hash_of(&format!("{value:?}"))
}

/// An argument as the canonicalizer sees it: either a child AST or an opaque
/// scalar such as an `Extract` bound.
enum CanonArg {
    Child,
    Scalar(u64),
}

/// The node's arguments in the same order and shape `Base.args` exposes them,
/// with non-AST entries reduced to a hash and children left as placeholders to
/// be filled from the traversal's child results.
///
/// The ordering matters: positions become part of the variable paths, and the
/// scalars become part of the hash, which is what distinguishes `Extract(7, 0,
/// x)` from `Extract(15, 8, x)`.
fn canon_args(ast: &AstRef<'static>) -> Vec<CanonArg> {
    match ast.op() {
        AstOp::ZeroExt(_, amount) | AstOp::SignExt(_, amount) => {
            vec![CanonArg::Scalar(hash_of(amount)), CanonArg::Child]
        }
        AstOp::Extract(_, end, start) => vec![
            CanonArg::Scalar(hash_of(end)),
            CanonArg::Scalar(hash_of(start)),
            CanonArg::Child,
        ],
        AstOp::FpAdd(_, _, rm)
        | AstOp::FpSub(_, _, rm)
        | AstOp::FpMul(_, _, rm)
        | AstOp::FpDiv(_, _, rm) => {
            vec![
                CanonArg::Child,
                CanonArg::Child,
                CanonArg::Scalar(hash_debug(rm)),
            ]
        }
        AstOp::FpSqrt(_, rm) => vec![CanonArg::Child, CanonArg::Scalar(hash_debug(rm))],
        _ => ast.child_iter().map(|_| CanonArg::Child).collect(),
    }
}

/// Hash of a leaf. For a symbol the name is dropped and only the sort is kept;
/// for a literal the value is kept.
fn leaf_hash(ast: &AstRef<'static>, op: &str) -> u64 {
    match ast.op() {
        AstOp::BVS(_, size) => hash_of(&(op, size)),
        AstOp::BoolS(_) | AstOp::StringS(_) => hash_of(&op),
        AstOp::FPS(_, sort) => hash_of(&(op, hash_debug(sort))),
        AstOp::BVV(value) => hash_of(&(op, value.to_biguint().to_bytes_le(), value.len())),
        AstOp::BoolV(value) => hash_of(&(op, value)),
        AstOp::FPV(value) => hash_of(&(op, hash_debug(value))),
        AstOp::StringV(value) => hash_of(&(op, value)),
        _ => hash_of(&op),
    }
}

// ----------------------------------------------------------- canonical hash

/// One step of the route from a node down to a variable: the op string, and the
/// operand position -- dropped for commutative ops so operand order stops
/// mattering, kept otherwise so position still distinguishes.
type Path = Vec<(String, Option<usize>)>;

/// How many times each path reaches one variable.
type PathCounter = BTreeMap<Path, u64>;

struct Canon {
    hash: u64,
    /// Variable node hash -> its paths, in first-seen order. The order is part
    /// of the algorithm: a parent reads its children's variables in this order
    /// when building its own sort keys.
    paths: Vec<(u64, PathCounter)>,
}

/// Memo for [`canonical`], keyed by AST hash. Sound without invalidation
/// because clarirs nodes are interned and immutable: one hash is always one
/// node. `walk` reads and fills it. It is never cleared: VeriBin analyses one
/// function per process, so the memo dies with the process.
static CANON_CACHE: LazyLock<GenericCache<u64, Arc<Canon>>> = LazyLock::new(GenericCache::default);

/// One node's canonical form, given its children's. Called by [`walk`] with the
/// children already done, in `child_iter` order.
fn canon_node(node: &AstRef<'static>, child_canon: &[Arc<Canon>]) -> Arc<Canon> {
    let op = node.to_opstring();

    if is_symbol_creation(&op) {
        let mut counter = PathCounter::new();
        counter.insert(vec![(op.clone(), Some(0))], 1);
        return Arc::new(Canon {
            hash: leaf_hash(node, &op),
            paths: vec![(node.hash(), counter)],
        });
    }
    if is_value_creation(&op) {
        return Arc::new(Canon {
            hash: leaf_hash(node, &op),
            paths: Vec::new(),
        });
    }

    // Pair each argument position with its child's canonical form, if it has
    // one, before any reordering.
    let mut children = child_canon.iter();
    let mut args: Vec<(u64, Option<Arc<Canon>>)> = canon_args(node)
        .into_iter()
        .map(|arg| match arg {
            CanonArg::Child => {
                let canon = children.next().expect("one canon per child").clone();
                (canon.hash, Some(canon))
            }
            CanonArg::Scalar(scalar) => (scalar, None),
        })
        .collect();

    // A reversible comparison is hashed as though it had been written the other
    // way round, so `a < b` and `b > a` land on the same hash.
    let reversed = reversed_op(&op);
    let effective_op = reversed.unwrap_or(&op).to_string();
    if reversed.is_some() {
        args.reverse();
    }
    let commutative = is_commutative(&effective_op);

    // Extend every child's paths with the step from here, merging the
    // children's variables into one ordered map.
    let mut order: Vec<u64> = Vec::new();
    let mut paths: HashMap<u64, PathCounter> = HashMap::new();
    for (index, (_, canon)) in args.iter().enumerate() {
        let Some(canon) = canon else { continue };
        let step = (
            effective_op.clone(),
            if commutative { None } else { Some(index) },
        );
        for (var, counter) in &canon.paths {
            let entry = paths.entry(*var).or_insert_with(|| {
                order.push(*var);
                PathCounter::new()
            });
            for (path, count) in counter {
                let mut extended = Vec::with_capacity(path.len() + 1);
                extended.push(step.clone());
                extended.extend(path.iter().cloned());
                *entry.entry(extended).or_insert(0) += count;
            }
        }
    }

    // Sort the operands by content, so a commutative op written either way round
    // hashes the same. Every operand carries its own hash in its sort key, so
    // the order is total up to genuinely identical operands.
    let mut keys: Vec<(u64, Option<Vec<PathCounter>>)> = args
        .iter()
        .map(|(hash, canon)| {
            let parent_counters = canon
                .as_ref()
                .map(|canon| canon.paths.iter().map(|(var, _)| paths[var].clone()).collect());
            (*hash, parent_counters)
        })
        .collect();
    keys.sort();
    let sorted_hashes: Vec<u64> = keys.iter().map(|(hash, _)| *hash).collect();

    // The variable paths, stripped of which variable they belong to: that is
    // what makes the hash blind to variable names but not to how often and
    // where each distinct variable is used.
    let mut nameless: Vec<Vec<(Path, u64)>> = order
        .iter()
        .map(|var| paths[var].iter().map(|(p, c)| (p.clone(), *c)).collect())
        .collect();
    nameless.sort();

    Arc::new(Canon {
        hash: hash_of(&(sorted_hashes, nameless)),
        paths: order
            .into_iter()
            .map(|var| {
                let counter = paths.remove(&var).expect("ordered var is in paths");
                (var, counter)
            })
            .collect(),
    })
}

fn canonical(ast: &AstRef<'static>) -> Result<Arc<Canon>, ClaripyError> {
    Ok(walk(
        ast.clone(),
        |_| Ok(None),
        |node, child_canon: &[Arc<Canon>]| Ok(canon_node(&node, child_canon)),
        &*CANON_CACHE,
    )?)
}

pub fn canonical_hash(ast: &AstRef<'static>) -> Result<u64, ClaripyError> {
    Ok(canonical(ast)?.hash)
}

// --------------------------------------------------------- canonicalization

/// Rename every symbolic leaf `var_0`, `var_1`, ... in traversal order.
///
/// Numbered per occurrence, not per distinct variable -- see the module notes.
/// That is why this walks uncached (`&()`): a shared subexpression must be
/// renumbered at each of its occurrences, not reused.
fn normalize_names(ast: &AstRef<'static>) -> Result<AstRef<'static>, ClaripyError> {
    let mut counter = 0usize;
    Ok(walk(
        ast.clone(),
        |_| Ok(None),
        |node, children: &[AstRef<'static>]| {
            if is_symbol_creation(&node.to_opstring()) {
                let name = format!("var_{counter}");
                counter += 1;
                return match node.op() {
                    AstOp::BVS(_, size) => GLOBAL_CONTEXT.bvs(name, *size),
                    AstOp::BoolS(_) => GLOBAL_CONTEXT.bools(name),
                    AstOp::FPS(_, sort) => GLOBAL_CONTEXT.fps(name, *sort),
                    AstOp::StringS(_) => GLOBAL_CONTEXT.strings(name),
                    _ => Ok(node.clone()),
                };
            }
            reconstruct_node(&GLOBAL_CONTEXT, &node, children)
        },
        &(),
    )?)
}

/// Sort commutative operands, flip reversible comparisons, optionally rename.
///
/// Only commutative operands are canonicalized recursively; everything else is
/// left as written. That is the original behaviour and the reason this is a
/// filter rather than a decision procedure.
pub fn canonicalize(ast: &AstRef<'static>, rename: bool) -> Result<AstRef<'static>, ClaripyError> {
    let op = ast.to_opstring();
    let mut new = ast.clone();

    if !is_symbol_creation(&op) && !is_value_creation(&op) {
        if let Some(reversed) = reverse(&new)? {
            new = reversed;
        }
        if is_commutative(&op) {
            let mut children = new
                .child_iter()
                .map(|child| {
                    let child = canonicalize(&child, false)?;
                    Ok((canonical(&child)?.hash, child))
                })
                .collect::<Result<Vec<_>, ClaripyError>>()?;
            children.sort_by_key(|(hash, _)| *hash);
            let children: Vec<_> = children.into_iter().map(|(_, child)| child).collect();
            new = reconstruct_node(&GLOBAL_CONTEXT, &new, &children)?;
        }
    }

    if rename {
        new = normalize_names(&new)?;
    }
    Ok(new)
}

/// Rebuild `ast` as `op` over `args`, restoring claripy's `make_like`.
///
/// claripy could build any op from its string. clarirs' constructors reject most
/// op strings, so this covers the three shapes VeriBin actually asks for and
/// raises on anything else rather than quietly producing a different node:
///
/// * renaming a symbolic leaf (`args[0]` is the new name) -- this is how
///   VeriBin strips the uniquifying id out of `reg_rdi_12_64`;
/// * rebuilding the same op over new children, keeping non-child fields;
/// * flipping a reversible comparison.
pub fn make_like<'py>(
    py: Python<'py>,
    ast: &AstRef<'static>,
    op: &str,
    args: Vec<Bound<'py, PyAny>>,
) -> Result<Bound<'py, Base>, ClaripyError> {
    let own_op = ast.to_opstring();

    if is_symbol_creation(&own_op) {
        if op != own_op {
            return Err(ClaripyError::TypeError(format!(
                "make_like cannot turn a {own_op} leaf into {op}"
            )));
        }
        let name: String = args
            .first()
            .ok_or_else(|| ClaripyError::TypeError("make_like on a leaf needs a name".into()))?
            .extract()
            .map_err(|_| ClaripyError::TypeError("make_like on a leaf needs a name".into()))?;
        let renamed = match ast.op() {
            AstOp::BVS(_, size) => GLOBAL_CONTEXT.bvs(name, *size)?,
            AstOp::BoolS(_) => GLOBAL_CONTEXT.bools(name)?,
            AstOp::FPS(_, sort) => GLOBAL_CONTEXT.fps(name, *sort)?,
            AstOp::StringS(_) => GLOBAL_CONTEXT.strings(name)?,
            _ => ast.clone(),
        };
        return Base::from_ast(py, renamed);
    }

    let children: Vec<AstRef<'static>> = args
        .iter()
        .filter_map(|arg| arg.cast::<Base>().ok().map(|b| b.get().ast()))
        .collect();

    if op == own_op {
        return Base::from_ast(py, reconstruct_node(&GLOBAL_CONTEXT, ast, &children)?);
    }

    if reversed_op(&own_op) == Some(op) {
        // The caller already handed us the swapped operands.
        let (a, b) = match children.as_slice() {
            [a, b] => (a.clone(), b.clone()),
            _ => {
                return Err(ClaripyError::TypeError(format!(
                    "make_like {op} needs two operands"
                )));
            }
        };
        let flipped = match op {
            "UGT" => AstOp::UGT(a, b),
            "UGE" => AstOp::UGE(a, b),
            "SGT" => AstOp::SGT(a, b),
            "SGE" => AstOp::SGE(a, b),
            _ => unreachable!("reversed_op only yields these four"),
        };
        return Base::from_ast(py, GLOBAL_CONTEXT.make_ast(flipped)?);
    }

    Err(ClaripyError::TypeError(format!(
        "make_like cannot build op {op} from {own_op}"
    )))
}

// --------------------------------------------------------------- cache keys

/// Hashable handle for an AST, as claripy's `cache_key` returned.
///
/// A clarirs node is already a sound dict key -- interned and value-hashed -- so
/// this is not needed for keying. It exists because VeriBin stores cache keys
/// and reads the AST back off them as `key.ast` in about eighteen places. The
/// AST is held directly rather than the Python wrapper, so the wrapper cache can
/// still evict; `.ast` re-wraps, which returns the same object if one is alive.
#[pyclass(frozen, module = "angr.rustylib.claripy.ast.base")]
pub struct ASTCacheKey {
    inner: AstRef<'static>,
}

impl ASTCacheKey {
    pub fn new(inner: AstRef<'static>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl ASTCacheKey {
    #[getter]
    pub fn ast<'py>(&self, py: Python<'py>) -> Result<Bound<'py, Base>, ClaripyError> {
        Base::from_ast(py, self.inner.clone())
    }

    pub fn __hash__(&self) -> usize {
        self.inner.hash() as usize
    }

    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        // Interning makes hash equality exactly structural equality here.
        match other.cast::<ASTCacheKey>() {
            Ok(other) => other.get().inner.hash() == self.inner.hash(),
            Err(_) => false,
        }
    }

    pub fn __repr__(&self, py: Python<'_>) -> Result<String, ClaripyError> {
        Ok(format!(
            "<Key {}>",
            Base::from_ast(py, self.inner.clone())?.get().__repr__()
        ))
    }
}
