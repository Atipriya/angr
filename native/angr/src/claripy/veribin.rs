//! Expression comparison helpers for VeriBin.
//!
//! VeriBin checks whether two decompiled functions match by comparing symbolic
//! expressions. Before calling the solver it asks a cheaper question: do the two
//! expressions have the same shape, ignoring variable names, the order of
//! operands where order does not matter, and whether a comparison was written
//! `a < b` or `b > a`? [`canonical_hash`] answers that. [`canonicalize`]
//! rewrites an expression so that two matching ones print the same.
//!
//! Traversal, rebuilding and caching come from clarirs ([`walk`],
//! [`reconstruct_node`], [`GenericCache`]); only the rules above are new here.
//! clarirs' own `algorithms::canonicalize` only renames variables.
//!
//! Two details are kept from the Python version on purpose, because changing
//! either changes which functions VeriBin reports as matching:
//!
//! * Renaming numbers each *use* of a variable rather than each variable, so
//!   `x + x` becomes `var_0 + var_1`, the same as `x + y`.
//! * The operator reaches the hash only through paths down to variables, so
//!   `1 + 2` and `1 * 2` hash the same.
//!
//! Hash values are dictionary keys within a single run and are never stored, so
//! they need not match the Python ones. Only which expressions share a hash
//! matters, and that does match.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, LazyLock};

use clarirs_core::algorithms::reconstruct::reconstruct_node;
use clarirs_core::algorithms::walk;
use clarirs_core::cache::GenericCache;

use crate::claripy::prelude::*;

// --------------------------------------------------------------- op classes
//
// Grouped by op string rather than by `AstOp`, because that is what the
// Python version used and the two differ: `Eq` on floats gives "fpEQ", which
// is not in the set below, while `Eq` on bitvectors gives "__eq__", which is.

/// Ops that create a variable. The name is dropped when hashing, which is what
/// lets two variables of the same type with different names compare equal.
fn is_symbol_creation(op: &str) -> bool {
    matches!(op, "BVS" | "BoolS" | "FPS" | "StringS")
}

/// Ops that create a constant.
fn is_value_creation(op: &str) -> bool {
    matches!(op, "BVV" | "BoolV" | "FPV" | "StringV")
}

/// Ops where operand order does not matter. `__eq__` and `__ne__` are VeriBin
/// additions: they count as unordered here, but clarirs' own simplifier is not
/// told so, because it is shared with angr's symbolic execution.
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
/// "less" forms are listed, so the result always uses the "greater" form.
/// claripy's table also had `__lt__`, `__radd__` and similar; clarirs never
/// produces those op strings, so they are left out.
fn reversed_op(op: &str) -> Option<&'static str> {
    match op {
        "ULT" => Some("UGT"),
        "ULE" => Some("UGE"),
        "SLT" => Some("SGT"),
        "SLE" => Some("SGE"),
        _ => None,
    }
}

/// The swapped node itself, for [`canonicalize`]. Matches [`reversed_op`].
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

/// Hash of a value that has no `Hash` but is fully described by its debug
/// output: float types, rounding modes and float constants.
fn hash_debug<T: std::fmt::Debug>(value: &T) -> u64 {
    hash_of(&format!("{value:?}"))
}

/// An argument: either a child expression or a plain number such as one of
/// `Extract`'s bit positions.
enum CanonArg {
    Child,
    Scalar(u64),
}

/// The node's arguments in the same order as `Base.args`, with numbers reduced
/// to a hash and children left as placeholders that the traversal fills in.
///
/// The order matters: positions become part of the paths below, and the numbers
/// become part of the hash, which is what tells `Extract(7, 0, x)` apart from
/// `Extract(15, 8, x)`.
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

/// Hash of a leaf: for a variable only its type, for a constant its value.
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

/// One step on the way down to a variable: the op string and which operand was
/// taken. The position is dropped where operand order does not matter, and kept
/// otherwise so that position still counts.
type Path = Vec<(String, Option<usize>)>;

/// How many times each path leads to one variable.
type PathCounter = BTreeMap<Path, u64>;

struct Canon {
    hash: u64,
    /// Variable hash to its paths, in the order first seen. The order matters:
    /// a parent reads its children's variables in it when building sort keys.
    paths: Vec<(u64, PathCounter)>,
}

/// Cache for [`canonical`], keyed by expression hash. It never needs clearing:
/// clarirs expressions cannot change, so one hash is always one expression, and
/// VeriBin analyses one function per process.
static CANON_CACHE: LazyLock<GenericCache<u64, Arc<Canon>>> = LazyLock::new(GenericCache::default);

/// One node's result, given its children's. [`walk`] calls this with the
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

    // Match each argument to its child's result before anything is reordered.
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

    // Hash `a < b` as though it were `b > a`, so both give the same hash.
    let reversed = reversed_op(&op);
    let effective_op = reversed.unwrap_or(&op).to_string();
    if reversed.is_some() {
        args.reverse();
    }
    let commutative = is_commutative(&effective_op);

    // Add this node's step to every child's paths, merging them into one map.
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

    // Sort the operands by content, so an unordered op written either way round
    // hashes the same. Each sort key contains that operand's own hash, so only
    // genuinely identical operands can tie.
    let mut keys: Vec<(u64, Option<Vec<PathCounter>>)> = args
        .iter()
        .map(|(hash, canon)| {
            let parent_counters = canon.as_ref().map(|canon| {
                canon
                    .paths
                    .iter()
                    .map(|(var, _)| paths[var].clone())
                    .collect()
            });
            (*hash, parent_counters)
        })
        .collect();
    keys.sort();
    let sorted_hashes: Vec<u64> = keys.iter().map(|(hash, _)| *hash).collect();

    // The paths without the variable they belong to. This is what makes the
    // hash ignore variable names while still counting how often and where each
    // variable is used.
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

/// Rename every variable `var_0`, `var_1`, ... in the order they are reached.
///
/// Numbers each use, not each variable -- see the notes at the top of the file.
/// That is why the walk is uncached (`&()`): a shared subexpression has to be
/// renumbered at each use rather than reused.
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

/// Sort unordered operands, swap comparisons, and optionally rename variables.
///
/// Only unordered operands are rewritten recursively; everything else is left
/// as written. That matches the Python version, and it is why this is a quick
/// check rather than a decision procedure.
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
/// claripy could build any op from its name; clarirs' constructors cannot. This
/// covers the three cases VeriBin asks for and raises on anything else rather
/// than quietly building something different:
///
/// * renaming a variable (`args[0]` is the new name), which is how VeriBin
///   removes the id from a name like `reg_rdi_12_64`;
/// * rebuilding the same op over new children, keeping its other fields;
/// * swapping a comparison.
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
        // The caller has already swapped the operands.
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
/// A clarirs expression already works as a dictionary key, so this is not
/// needed for that. It exists because VeriBin stores these keys and reads the
/// expression back out as `key.ast` in about eighteen places. It holds the
/// expression rather than the Python object, so the Python object can still be
/// freed; `.ast` rebuilds it, returning the same one if it is still alive.
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
        // Equal hashes mean equal expressions, so this is exact.
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
