# Contributing to zmath

Thank you for your interest in contributing to zmath! This document provides guidelines and information for contributors.

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:
- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors

## How to Contribute

### Development Setup

1. **Prerequisites**
   - Zig 0.16.0-dev or later
   - Git

2. **Clone and Setup**
   ```bash
   git clone https://github.com/ghostkellz/zmath.git
   cd zmath
   zig build test  # Run tests to ensure everything works
   zig build run   # Run the demonstration
   ```

### Development Workflow

1. **Choose an Issue**: Look for issues labeled `good first issue` or `help wanted`
2. **Create a Branch**: `git checkout -b feature/your-feature-name`
3. **Make Changes**: Follow the coding standards below
4. **Test Thoroughly**: Run `zig build test` and add new tests if needed
5. **Commit**: Use clear, descriptive commit messages
6. **Push and Create PR**: Push your branch and create a pull request

### Coding Standards

#### Zig Style Guidelines
- Follow the [Zig Style Guide](https://ziglang.org/documentation/master/#Style-Guide)
- Use 4 spaces for indentation
- Use `camelCase` for variable and function names
- Use `PascalCase` for types and structs
- Maximum line length: 100 characters

#### Code Structure
```zig
// Good: Clear naming and documentation
/// Computes the dot product of two vectors.
/// Returns an error if dimensions don't match.
pub fn dot(a: Vector, b: Vector) !f64 {
    if (a.size() != b.size()) {
        return error.DimensionMismatch;
    }
    // Implementation...
}
```

#### Error Handling
- Use Zig's error union types for all fallible operations
- Provide descriptive error names (e.g., `DimensionMismatch`, `InvalidDimensions`)
- Document when functions can return errors

#### Memory Management
- Use the provided allocator parameter for all allocations
- Always call `deinit()` methods to free resources
- Follow RAII patterns with defer statements

#### Testing
- Write comprehensive tests for all public functions
- Test both success and error cases
- Use `std.testing.allocator` for memory leak detection
- Include edge cases (empty inputs, single elements, etc.)

### Types of Contributions

#### 🐛 Bug Fixes
- Fix reported bugs with minimal, targeted changes
- Include a test case that reproduces the bug
- Ensure the fix doesn't break existing functionality

#### ✨ New Features
- Discuss major features in issues before implementation
- Follow the existing API patterns and naming conventions
- Provide comprehensive documentation and examples
- Include performance considerations

#### 📚 Documentation
- Improve existing documentation
- Add code examples and usage patterns
- Fix typos and clarify confusing sections
- Update API documentation for new features

#### 🧪 Testing
- Add missing test coverage
- Improve test performance and reliability
- Add benchmark tests for performance-critical code

#### 🔧 Maintenance
- Update dependencies
- Improve build system and CI
- Refactor code for better maintainability
- Update documentation for breaking changes

### Pull Request Process

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what the PR does and why
3. **Checklist**:
   - [ ] Tests pass (`zig build test`)
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No memory leaks
   - [ ] Performance not regressed

4. **Review**: Address reviewer feedback promptly
5. **Merge**: Maintainers will merge approved PRs

### Commit Guidelines

- Use present tense in commit messages ("Add feature" not "Added feature")
- Keep commits focused on single changes
- Reference issue numbers when applicable
- Use `fix:`, `feat:`, `docs:`, `test:`, `refactor:` prefixes

### Areas for Contribution

#### High Priority
- Advanced linear algebra algorithms (LU/QD decomposition)
- Performance optimizations (SIMD, parallel processing)
- Extended mathematical functions

#### Medium Priority
- Sparse matrix support
- Complex number operations
- GPU acceleration framework

#### Low Priority
- Additional statistical distributions
- Machine learning primitives
- Visualization tools

### Getting Help

- **Issues**: Report bugs or request features
- **Discussions**: Ask questions or discuss ideas
- **Discord**: Join the community chat (link in README)

### Recognition

Contributors are recognized in:
- Git history and commit messages
- CHANGELOG.md for significant contributions
- README acknowledgments for major contributors

Thank you for contributing to zmath! 🚀