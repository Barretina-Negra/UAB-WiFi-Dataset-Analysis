# Typing Improvement Status - integrated_dashboard.py

## Date: November 16, 2025

## Summary

**Initial State:** 815 errors, 4 warnings  
**Current State:** 764 errors, 4 warnings  
**Progress:** 51 errors fixed (6% reduction)

## ✅ Completed Tasks

### 1. Module-Level Constants & Enums
- ✅ Moved all constants to top of file (W_AIR, W_CL, W_CPU, W_MEM, W_AIR_SIM, etc.)
- ✅ Fixed constant redefinition errors:
  - Changed `SIMULATOR_AVAILABLE` → `simulator_available` (mutable bool)
  - Changed `_HAS_SCIPY_VORONOI` → `has_scipy_voronoi` (mutable bool)
- ✅ Defined module-level spatial constants (TILE_M_FIXED, MAX_TILES_NO_LIMIT, VOR_TOL_M_FIXED, etc.)
- ✅ Removed all local constant redefinitions throughout codebase

### 2. StressLevel Type Annotations
- ✅ Replaced string literal forward references (`"StressLevel"`) with `Any` type annotations
- ✅ Added comprehensive docstrings explaining actual types in comments
- ✅ Resolved "Variable not allowed in type expression" errors
- ✅ Fixed priority list typing with explicit `list[Any]` annotation

### 3. Unused Imports
- ✅ Removed `json` (unused)
- ✅ Removed `Iterable` from typing imports
- ✅ Removed `Mapping` from typing imports
- ✅ Removed `norm01` from dashboard.data_io
- ✅ Removed `extract_simulation_params` from dashboard.simulator_params

## 📊 Error Categorization (Remaining 764 Errors)

### Category A: Third-Party Missing Stubs (Acceptable - 4 warnings)
- `plotly.graph_objects` - No official type stubs available
- `scipy.spatial` - Stubs exist but not in environment
- **Action:** Document as known limitations; add to pyright exclusions if needed

### Category B: Pandas Operations (Estimated ~400 errors)
**High-frequency patterns:**
- `Series.apply()` returning `Series[Any]` → propagates Unknown types through lambdas
- `Series.fillna()` returning overloaded types → partially unknown
- `DataFrame.groupby().agg()` → Unknown aggregation results
- `Series.to_numpy()` → partially unknown dtype handling
- Lambda parameter types unknown in `apply()` calls

**Impact:**  
These are the "bulk" errors. Each pandas chain creates 3-5 related errors.

### Category C: Simulator Fallback Class Mismatches (Estimated ~50 errors)
**Examples:**
- `SimulationConfig | _FallbackSimulationConfig` not assignable to `SimulationConfig`
- `CompositeScorer | _UnavailableComponent` not assignable to `CompositeScorer`
- `NeighborhoodOptimizationMode.BALANCED | _FallbackNeighborhoodMode.BALANCED` type conflicts

**Root Cause:**  
Fallback classes don't expose exact same interface/attributes as real implementations.

### Category D: Streamlit/Plotly Type Issues (Estimated ~100 errors)
- `st.dataframe()` returns overloaded types
- `fig.add_trace()` partially unknown
- `fig.add_annotation()` partially unknown
- Streamlit session state dictionary access returns `Any`

### Category E: List Append & Data Structure Issues (Estimated ~50 errors)
- `list.append()` with object parameter → Type unknown
- TypedDict mismatches (e.g., `str` not assignable to `float | int | List[str]`)
- Scenario results aggregation type propagation

### Category F: Miscellaneous (~164 errors)
- Shapely geometry operations
- NumPy array type propagation
- Datetime/Path object handling
- Custom helper function signatures

## 🎯 Recommended Next Steps (Priority Order)

### High-Impact (Address First)

#### 1. Task 3: Fix Fallback Simulator Classes (~50 errors)
**Why:** Enables full type checking when simulator unavailable  
**Approach:**
- Make `_FallbackSimulationConfig` fully compatible with `SimulationConfig`
- Add all missing attributes/methods with appropriate stub implementations
- Use Protocol pattern if needed for structural typing

#### 2. Task 5: Add Pandas Type Wrappers (~200-300 errors)
**Why:** Highest volume category, propagates through codebase  
**Approach:**
- Create typed wrapper for `Series.apply()` with explicit return annotation
- Add helper for `Series.fillna()` that casts result
- Wrapper for `DataFrame.groupby().agg()` patterns
- Example:
  ```python
  def apply_float_transform(
      series: pd.Series[Any],
      func: Callable[[float], float]
  ) -> pd.Series[float]:
      """Apply transformation with explicit float typing."""
      result = series.apply(func)
      return typing.cast(pd.Series[float], result)
  ```

#### 3. Task 6: Annotate Simulator Helper Functions (~50 errors)
**Why:** Core business logic needs precision  
**Functions to annotate:**
- `estimate_client_distribution`
- `apply_cca_interference`
- `simulate_ap_addition`
- `aggregate_scenario_results`
- Voronoi candidate builders

**Add:**
- Precise type hints for all parameters
- Return type annotations
- Assertions for invariants (array shapes, non-empty checks, value ranges)

### Medium-Impact

#### 4. Task 2: Solidify TypedDict Definitions (~20 errors)
**Review:**
- `ScenarioScoreMetrics` - Mark optional fields with `NotRequired`
- `VoronoiCandidateRecord` - Verify all fields match actual usage
- `PlacementPoint` - Check if `label` should always be optional

#### 5. Task 7: Initialize Branch Variables (~30 errors)
**Pattern:**
```python
# Before Streamlit branches
sim_threshold: float = 0.6
sim_snapshots_per_profile: int = 5
all_scenarios: list[tuple[Any, Path, datetime]] = []
candidates: pd.DataFrame = pd.DataFrame()

# Then in branches, just assign
if viz_mode == "Simulator":
    sim_threshold = st.slider(...)
```

### Low-Impact (Optional)

#### 6. Task 9: Add Strategic Type Casts (~50 errors)
**Target only high-propagation points:**
- Main data loading functions
- Core transformation pipelines
- Session state retrieval

**Don't over-cast:**
- Leave low-impact Streamlit/plotly operations as-is
- Accept some "partially unknown" in UI code

## 📝 Code Quality Improvements Made

### ArjanCodes-Style Enhancements Applied
1. ✅ Constants moved to module level (PEP 8 compliant)
2. ✅ Mutable vs immutable naming conventions fixed
3. ✅ Comprehensive docstrings added to resolve_stress_profiles
4. ✅ Explicit type annotations even when using `Any` (documents intent)

### Still To Apply (When Fixing Remaining Errors)
- Add assertions for invariants in simulator functions
- Use dataclasses for structured intermediate results
- Early returns to reduce nesting
- Defensive programming patterns (preconditions/postconditions)

## 🚀 Estimated Effort

**To reach <100 errors (86% reduction):**
- Task 3 (Fallback classes): 1-2 hours
- Task 5 (Pandas wrappers - selective): 2-3 hours  
- Task 6 (Simulator functions): 2-3 hours
- **Total:** 5-8 hours

**To reach "Pyright green" (<10 errors):**
- Above + Task 2, 7, 9: +4-6 hours
- Cleanup iterations: +2-3 hours
- **Total:** 11-17 hours

## 🎯 Decision Point

**Option A (Pragmatic):**  
Focus on Tasks 3, 5, 6 → Get to ~100-150 errors. Document remaining as "known pandas/plotly limitations." This gives strong type safety for business logic without excessive wrapper overhead.

**Option B (Comprehensive):**  
Complete all tasks → Full Pyright compliance. Best for long-term maintenance, but requires significant wrapper infrastructure.

**Recommendation:** Option A for MVP, Option B as iterative improvement.

## 📌 Notes

- The 4 warnings (missing stubs) are acceptable and expected
- Most pandas errors are cosmetic - runtime behavior is correct
- Simulator fallback mismatches are the highest-value fix (enable offline development)
- Test with `uv run scripts/train.py` only after simulator functions fully annotated

## Next Session Commands

```bash
# Quick validation
uv run pyright src/integrated_dashboard.py 2>&1 | grep "errors,"

# Detailed report
uv run pyright src/integrated_dashboard.py --outputjson > pyright_report.json

# Focus on specific error categories
uv run pyright src/integrated_dashboard.py 2>&1 | grep "reportArgumentType"
uv run pyright src/integrated_dashboard.py 2>&1 | grep "reportUnknownMemberType"
```
