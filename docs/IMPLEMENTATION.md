# zmath Implementation Guide

## Overview

This document outlines the implementation details of the zmath library, covering the architecture decisions, data structures, and algorithms used in our first iteration.

## Architecture

### Core Design Principles

1. **Memory Safety**: All operations use Zig's allocator system for proper memory management
2. **Type Safety**: Compile-time error checking for dimension mismatches
3. **Performance**: Zero-cost abstractions with minimal runtime overhead
4. **Ergonomics**: Clean, idiomatic Zig interfaces

### Module Structure

```
src/
├── root.zig          # Main library module
└── main.zig          # Demonstration program
```

## Data Structures

### Vector

```zig
pub const Vector = struct {
    data: []f64,              // Contiguous memory for elements
    allocator: std.mem.Allocator,  // Memory management
};
```

**Key Features:**
- Dynamic allocation with proper cleanup via `deinit()`
- Bounds checking through Zig's built-in safety
- f64 precision for mathematical accuracy

### Matrix

```zig
pub const Matrix = struct {
    data: []f64,              // Row-major storage
    rows: usize,              // Number of rows
    cols: usize,              // Number of columns
    allocator: std.mem.Allocator,  // Memory management
};
```

**Storage Format:** Row-major order for cache efficiency
- Element at (i,j) stored at index: `i * cols + j`
- Contiguous row storage improves memory locality

## Algorithms Implemented

### Linear Algebra Operations

#### Vector Operations

**Addition/Subtraction:**
```zig
for (0..a.size()) |i| {
    result.set(i, a.get(i) + b.get(i));  // Element-wise operation
}
```
- Time Complexity: O(n)
- Space Complexity: O(n) for result vector

**Dot Product:**
```zig
var sum: f64 = 0.0;
for (0..a.size()) |i| {
    sum += a.get(i) * b.get(i);
}
```
- Time Complexity: O(n)
- Space Complexity: O(1)

**Magnitude:**
```zig
var sum: f64 = 0.0;
for (0..a.size()) |i| {
    const val = a.get(i);
    sum += val * val;
}
return @sqrt(sum);
```
- Uses L2 norm (Euclidean distance)
- Time Complexity: O(n)

#### Matrix Operations

**Matrix Multiplication:**
```zig
for (0..a.rows) |i| {
    for (0..b.cols) |j| {
        var sum: f64 = 0.0;
        for (0..a.cols) |k| {
            sum += a.get(i, k) * b.get(k, j);
        }
        result.set(i, j, sum);
    }
}
```
- Standard O(n³) algorithm
- Could be optimized with blocking, SIMD, or Strassen's algorithm in future

**Transpose:**
```zig
for (0..a.rows) |i| {
    for (0..a.cols) |j| {
        result.set(j, i, a.get(i, j));  // Swap indices
    }
}
```
- Time Complexity: O(m×n)
- Creates new matrix (out-of-place operation)

### Statistical Functions

#### Mean
```zig
var sum: f64 = 0.0;
for (data) |value| {
    sum += value;
}
return sum / @as(f64, @floatFromInt(data.len));
```
- Single-pass algorithm
- Time Complexity: O(n)

#### Variance
```zig
const m = mean(data);
var sum_sq_diff: f64 = 0.0;
for (data) |value| {
    const diff = value - m;
    sum_sq_diff += diff * diff;
}
return sum_sq_diff / @as(f64, @floatFromInt(data.len));
```
- Two-pass algorithm (first for mean, second for variance)
- Uses population variance formula
- Time Complexity: O(n)

## Error Handling

### Dimension Checking

```zig
if (a.size() != b.size()) {
    return error.DimensionMismatch;
}
```

All operations validate dimensions at runtime and return appropriate errors:
- `DimensionMismatch`: Vector/matrix dimensions incompatible
- `InvalidDimensions`: Matrix creation with mismatched data

### Memory Management

All allocation operations use Zig's allocator pattern:
```zig
const data = try allocator.alloc(f64, size);
// ... use data ...
defer allocator.free(data);  // Automatic cleanup
```

## Performance Characteristics

### Memory Layout
- **Vectors**: Contiguous f64 array
- **Matrices**: Row-major contiguous f64 array
- **Cache Performance**: Good for row-wise operations

### Time Complexities
| Operation | Complexity |
|-----------|------------|
| Vector Add/Sub | O(n) |
| Vector Dot Product | O(n) |
| Matrix Add/Sub | O(m×n) |
| Matrix Multiply | O(m×n×p) |
| Matrix Transpose | O(m×n) |
| Statistical Functions | O(n) |

## Future Optimizations

### Phase 2 Planned Improvements

1. **SIMD Vectorization**
   - Use Zig's vector types for parallel operations
   - Target AVX/SSE on x86, NEON on ARM

2. **Blocked Matrix Multiplication**
   - Improve cache locality for large matrices
   - Reduce memory bandwidth requirements

3. **In-place Operations**
   - Add variants that modify existing matrices
   - Reduce memory allocation overhead

4. **Sparse Matrix Support**
   - CSR/CSC formats for sparse matrices
   - Specialized algorithms for sparse operations

### Phase 3 Advanced Features

1. **GPU Acceleration**
   - CUDA/OpenCL backends
   - Automatic CPU/GPU dispatch

2. **Advanced Decompositions**
   - LU, QR, SVD decompositions
   - Eigenvalue/eigenvector computation

## Testing Strategy

### Test Coverage
- Unit tests for all public functions
- Error condition testing
- Memory leak prevention
- Edge case handling (empty data, single elements)

### Test Categories
1. **Functionality Tests**: Verify mathematical correctness
2. **Error Tests**: Ensure proper error handling
3. **Memory Tests**: Check for leaks and proper cleanup
4. **Performance Tests**: Future benchmark suite

## Dependencies

### Standard Library Usage
- `std.mem.Allocator`: Memory management
- `std.testing`: Test framework
- `std.debug`: Demonstration output
- `std.math`: Mathematical functions (@sqrt)

### Zero External Dependencies
- Pure Zig implementation
- No C library dependencies
- Self-contained mathematical operations