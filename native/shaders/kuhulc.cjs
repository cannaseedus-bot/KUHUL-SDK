#!/usr/bin/env node

/**
 * kuhulc.js - K'uhul Compiler: Lexer → Parser → Semantic Analyzer → KSON Generator
 * Phases 1-4 of 6-phase compiler architecture
 * Version: 1.0.0
 */

const fs = require('fs');
const path = require('path');

// ============================================================================
// ERRORS
// ============================================================================

class KuhulSyntaxError extends Error {
  constructor(message, line = 0, col = 0, context = '') {
    super(message);
    this.name = 'KuhulSyntaxError';
    this.line = line;
    this.col = col;
    this.context = context;
  }

  toString() {
    const locStr = this.line > 0 ? ` (Line ${this.line}${this.col > 0 ? `, Col ${this.col}` : ''})` : '';
    const ctxStr = this.context ? `\n  Context: ${this.context}` : '';
    return `${this.name}: ${this.message}${locStr}${ctxStr}`;
  }
}

class SemanticError extends Error {
  constructor(errors) {
    const errorList = Array.isArray(errors) ? errors : [errors];
    super(errorList.join('\n'));
    this.name = 'SemanticError';
    this.errors = errorList;
  }

  toString() {
    return `${this.name}:\n${this.errors.map(e => `  - ${e}`).join('\n')}`;
  }
}

// ============================================================================
// PHASE 1: LEXER
// ============================================================================

class KuhulLexer {
  constructor() {
    this.source = '';
    this.pos = 0;
    this.line = 1;
    this.col = 1;
    this.tokens = [];
  }

  tokenize(source) {
    this.source = source;
    this.pos = 0;
    this.line = 1;
    this.col = 1;
    this.tokens = [];

    while (this.pos < this.source.length) {
      // Skip whitespace
      if (this.isWhitespace(this.current())) {
        this.skipWhitespace();
        continue;
      }

      // Skip comments
      if (this.current() === ';') {
        this.skipComment();
        continue;
      }

      // Try to match multi-character tokens
      let matched = false;

      // Operators first
      const opMatch = this.tryMatch([
        { pattern: /^→/, type: 'OP_PIPE', value: '→' },
        { pattern: /^⊗/, type: 'OP_TENSOR_DOT', value: '⊗' },
        { pattern: /^∫/, type: 'OP_INTEGRAL', value: '∫' },
        { pattern: /^∇/, type: 'OP_GRADIENT', value: '∇' },
        { pattern: /^⊙/, type: 'OP_ATTEND', value: '⊙' },
      ]);

      if (opMatch) {
        this.addToken(opMatch.type, opMatch.value);
        matched = true;
      }

      if (!matched) {
        // Numbers
        if (this.isDigit(this.current())) {
          this.scanNumber();
          matched = true;
        }
      }

      if (!matched) {
        // Strings
        if (this.current() === '"') {
          this.scanString();
          matched = true;
        }
      }

      if (!matched) {
        // Identifiers and keywords
        if (this.isIdentifierStart(this.current())) {
          this.scanIdentifier();
          matched = true;
        }
      }

      if (!matched) {
        // Single-character tokens
        const char = this.current();
        const singleCharTokens = {
          '(': 'LPAREN',
          ')': 'RPAREN',
          '[': 'LBRACKET',
          ']': 'RBRACKET',
          ',': 'COMMA',
          '=': 'EQ',
        };

        if (singleCharTokens[char]) {
          this.addToken(singleCharTokens[char], char);
          this.advance();
          matched = true;
        }
      }

      if (!matched) {
        throw new KuhulSyntaxError(
          `Unexpected character: '${this.current()}'`,
          this.line,
          this.col,
          `at position ${this.pos}`
        );
      }
    }

    this.addToken('EOF', '');
    return this.tokens;
  }

  tryMatch(patterns) {
    const remaining = this.source.slice(this.pos);
    for (const p of patterns) {
      const match = remaining.match(p.pattern);
      if (match) {
        const matched = {
          type: p.type,
          value: p.value || match[0],
          length: match[0].length,
        };
        this.advance(match[0].length);
        return matched;
      }
    }
    return null;
  }

  scanNumber() {
    const startCol = this.col;
    let value = '';
    while (this.isDigit(this.current())) {
      value += this.current();
      this.advance();
    }
    if (this.current() === '.' && this.isDigit(this.peek())) {
      value += this.current();
      this.advance();
      while (this.isDigit(this.current())) {
        value += this.current();
        this.advance();
      }
    }
    this.tokens.push({
      type: 'NUMBER',
      value: value,
      line: this.line,
      col: startCol,
    });
  }

  scanString() {
    const startCol = this.col;
    this.advance(); // skip opening quote
    let value = '';
    while (this.current() !== '"' && !this.isEOF()) {
      if (this.current() === '\\') {
        this.advance();
        value += this.current();
        this.advance();
      } else {
        value += this.current();
        this.advance();
      }
    }
    if (this.current() !== '"') {
      throw new KuhulSyntaxError('Unterminated string', this.line, startCol);
    }
    this.advance(); // skip closing quote
    this.tokens.push({
      type: 'STRING',
      value: value,
      line: this.line,
      col: startCol,
    });
  }

  scanIdentifier() {
    const startCol = this.col;
    let value = '';
    while (this.isIdentifierChar(this.current())) {
      value += this.current();
      this.advance();
    }
    this.tokens.push({
      type: 'IDENTIFIER',
      value: value,
      line: this.line,
      col: startCol,
    });
  }

  skipWhitespace() {
    while (this.isWhitespace(this.current())) {
      this.advance();
    }
  }

  skipComment() {
    while (this.current() !== '\n' && !this.isEOF()) {
      this.advance();
    }
  }

  addToken(type, value) {
    this.tokens.push({
      type,
      value,
      line: this.line,
      col: this.col,
    });
  }

  isWhitespace(char) {
    return /[\s]/.test(char);
  }

  isDigit(char) {
    return /[0-9]/.test(char);
  }

  isIdentifierStart(char) {
    return /[a-zA-Z_]/.test(char);
  }

  isIdentifierChar(char) {
    return /[a-zA-Z0-9_]/.test(char);
  }

  current() {
    return this.pos < this.source.length ? this.source[this.pos] : '';
  }

  peek(offset = 1) {
    return this.pos + offset < this.source.length ? this.source[this.pos + offset] : '';
  }

  advance(count = 1) {
    for (let i = 0; i < count; i++) {
      if (this.source[this.pos] === '\n') {
        this.line++;
        this.col = 1;
      } else {
        this.col++;
      }
      this.pos++;
    }
  }

  isEOF() {
    return this.pos >= this.source.length;
  }
}

// ============================================================================
// PHASE 2: PARSER
// ============================================================================

class KuhulParser {
  constructor(tokens) {
    this.tokens = tokens;
    this.pos = 0;
  }

  parse() {
    const statements = [];

    while (!this.isEOF()) {
      statements.push(this.parseStatement());
    }

    return {
      type: 'Program',
      statements,
    };
  }

  parseStatement() {
    if (!this.check('LBRACKET')) {
      throw new KuhulSyntaxError(
        `Expected '[' but got '${this.current().value}'`,
        this.current().line,
        this.current().col
      );
    }

    return this.parseGlyph();
  }

  parseGlyph() {
    this.expect('LBRACKET');

    // Get glyph type - should be a recognized glyph name
    const glyphToken = this.current();
    const glyphName = glyphToken.value;
    
    const glyphType = glyphName;
    this.advance();

    // Parse until closing bracket
    let name = '';
    const args = [];
    const body = [];
    let hasArgs = false;

    while (!this.check('RBRACKET')) {
      // Check if it's a nested glyph (any [ identifier ... ])
      if (this.check('LBRACKET')) {
        const nextToken = this.peek();
        if (nextToken && nextToken.type === 'IDENTIFIER') {
          body.push(this.parseGlyph());
          continue;
        }
      }

      // Check if this could be a name (first identifier, before any args)
      // Name is only assigned if:
      // 1. We haven't assigned a name yet
      // 2. Current token is an IDENTIFIER
      // 3. We haven't parsed any arguments yet
      // 4. Next token is not '=' (which would make it a parameter name)
      // 5. Next token is not an operator (which would make current a name)
      if (name === '' && !hasArgs && this.current().type === 'IDENTIFIER' && 
          !this.isOperator(this.peek()) && this.peek().type !== 'EQ') {
        // This is a name
        name = this.current().value;
        this.advance();
        continue;
      }

      // Parse argument
      const arg = this.parseArg();
      args.push(arg);
      hasArgs = true;
      
      // Skip comma if present and next token is not ] and not a glyph
      // But not if we just parsed a shape (which consumes its own commas)
      if (this.check('COMMA') && arg.key !== 'shape' && this.peekAt(this.pos + 1).type !== 'RBRACKET') {
        this.advance();
      }
    }

    this.expect('RBRACKET');

    return {
      type: 'Glyph',
      glyph: glyphType,
      glyphValue: glyphType,
      name,
      args,
      body,
      line: glyphToken.line,
      col: glyphToken.col,
    };
  }

  isOperator(token) {
    return token && [
      'OP_PIPE',
      'OP_TENSOR_DOT',
      'OP_INTEGRAL',
      'OP_GRADIENT',
      'OP_ATTEND',
    ].includes(token.type);
  }

  parseArg() {
    const token = this.current();

    if (token.type === 'IDENTIFIER') {
      const key = token.value;
      this.advance();

      if (this.check('EQ')) {
        this.advance();
        const valueToken = this.current();
        let value = valueToken.value;

        if (valueToken.type === 'NUMBER') {
          value = parseFloat(value);
        }

        this.advance();

        // Handle tuples for shape=(N,M,...)
        if (key === 'shape' && this.check('COMMA')) {
          // Accumulate all comma-separated values
          let tupleStr = String(value);
          while (this.check('COMMA')) {
            this.advance();  // skip comma
            if (this.current().type === 'NUMBER') {
              tupleStr += ',' + this.current().value;
              this.advance();
            } else {
              break;
            }
          }
          value = tupleStr;
        }

        return { key, value };
      } else {
        // Positional argument
        return { value: key };
      }
    }

    if (token.type === 'NUMBER') {
      const value = parseFloat(token.value);
      this.advance();
      return { value };
    }

    if (token.type === 'OP_TENSOR_DOT' || token.type === 'OP_GRADIENT' || 
        token.type === 'OP_INTEGRAL' || token.type === 'OP_ATTEND' || token.type === 'OP_PIPE') {
      const value = token.value;
      this.advance();
      return { value };
    }

    throw new KuhulSyntaxError(
      `Unexpected token in argument: '${token.value}'`,
      token.line,
      token.col
    );
  }

  check(type) {
    return this.current().type === type;
  }

  advance() {
    if (this.pos < this.tokens.length) {
      this.pos++;
    }
  }

  current() {
    return this.tokens[this.pos] || { type: 'EOF', value: '', line: 0, col: 0 };
  }

  peek(offset = 1) {
    return this.peekAt(this.pos + offset);
  }

  peekAt(index) {
    return this.tokens[index] || { type: 'EOF', value: '', line: 0, col: 0 };
  }

  expect(type) {
    if (this.current().type !== type) {
      throw new KuhulSyntaxError(
        `Expected '${type}' but got '${this.current().value}'`,
        this.current().line,
        this.current().col
      );
    }
    const token = this.current();
    this.advance();
    return token;
  }

  isEOF() {
    return this.current().type === 'EOF';
  }
}

// ============================================================================
// PHASE 3: SEMANTIC ANALYZER
// ============================================================================

class KuhulSemanticAnalyzer {
  constructor(ast) {
    this.ast = ast;
    this.symbolTable = new Map();
    this.errors = [];
    // Use actual glyph names as they appear in source
    this.validGlyphs = new Set([
      'Pop',
      'Yax',
      "Ch'en",
      'Sek',
      'Wo',
      "K'ayab'",
      "Kumk'u",
      'Muwan',
      'Xul',
      '∇',
      '∫',
      '⊙',
      '⊗',
      'Chen',  // Alternate form
    ]);
    this.validDtypes = new Set(['float32', 'float64', 'int32', 'int8', 'uint32']);
    this.validOperators = new Set(['⊗', '∫', '∇', '⊙', 'reduce', 'map']);
  }

  analyze() {
    for (const stmt of this.ast.statements) {
      this.analyzeStatement(stmt);
    }

    if (this.errors.length > 0) {
      throw new SemanticError(this.errors);
    }

    return this.ast;
  }

  analyzeStatement(stmt) {
    if (stmt.type !== 'Glyph') return;

    // Validate glyph is known
    if (!this.validGlyphs.has(stmt.glyph)) {
      this.error(`Unknown glyph: ${stmt.glyph}`, stmt.line);
      return;
    }

    switch (stmt.glyph) {
      case 'Pop':
        this.analyzeFunction(stmt);
        break;
      case 'Wo':
        this.analyzeTensor(stmt);
        break;
      case 'Yax':
        this.analyzeRead(stmt);
        break;
      case 'Chen':
      case "Ch'en":
        this.analyzeWrite(stmt);
        break;
      case 'Sek':
        this.analyzeExecute(stmt);
        break;
      case 'Xul':
        // Xul is a sync/end marker
        break;
      default:
        break;
    }

    // Analyze nested body
    if (stmt.body && stmt.body.length > 0) {
      for (const bodyStmt of stmt.body) {
        this.analyzeStatement(bodyStmt);
      }
    }
  }

  analyzeFunction(stmt) {
    // [Pop name] must have a body
    if (!stmt.body || stmt.body.length === 0) {
      this.error(`Function '${stmt.name}' must have a body`, stmt.line);
    }

    this.symbolTable.set(stmt.name, {
      type: 'function',
      name: stmt.name,
      line: stmt.line,
    });
  }

  analyzeTensor(stmt) {
    // [Wo name shape=(N,M) dtype=float32]
    const shape = this.getArg(stmt, 'shape');
    const dtype = this.getArg(stmt, 'dtype') || 'float32';

    if (!shape) {
      this.error(`Tensor '${stmt.name}' missing 'shape' argument`, stmt.line);
      return;
    }

    if (!this.validDtypes.has(dtype)) {
      this.error(`Invalid dtype: ${dtype} (valid: ${Array.from(this.validDtypes).join(', ')})`, stmt.line);
    }

    // Validate shape
    if (typeof shape !== 'string' || !shape.match(/^\d+(\s*,\s*\d+)*$/)) {
      this.error(`Invalid shape format: ${shape}`, stmt.line);
      return;
    }

    const shapeTuple = shape.split(',').map(s => parseInt(s.trim()));
    for (const dim of shapeTuple) {
      if (dim <= 0) {
        this.error(`Shape dimensions must be positive, got: ${dim}`, stmt.line);
      }
    }

    this.symbolTable.set(stmt.name, {
      type: 'tensor',
      name: stmt.name,
      shape: shapeTuple,
      dtype,
      line: stmt.line,
    });
  }

  analyzeRead(stmt) {
    if (!this.symbolTable.has(stmt.name)) {
      this.error(`Undefined variable: ${stmt.name}`, stmt.line);
    }
  }

  analyzeWrite(stmt) {
    if (!this.symbolTable.has(stmt.name)) {
      this.error(`Undefined variable: ${stmt.name}`, stmt.line);
    }
  }

  analyzeExecute(stmt) {
    // [Sek ⊗ A B] or similar
    // First positional arg should be operator
    const firstArg = stmt.args.find(a => !a.key);
    if (firstArg && !this.validOperators.has(firstArg.value)) {
      // Just warn, don't error - could be future operator
    }

    // Check that operands exist
    for (const arg of stmt.args) {
      if (arg.key && !this.symbolTable.has(arg.key)) {
        this.error(`Undefined variable in operation: ${arg.key}`, stmt.line);
      }
    }
  }

  getArg(stmt, key) {
    const arg = stmt.args.find(a => a.key === key);
    return arg ? arg.value : null;
  }

  error(msg, line) {
    this.errors.push(`Line ${line}: ${msg}`);
  }
}

// ============================================================================
// PHASE 4: KSON GENERATOR
// ============================================================================

class KSONGenerator {
  constructor(ast, manifest) {
    this.ast = ast;
    this.manifest = manifest || {
      name: 'kernel',
      type: 'compute_kernel',
      target: 'directx_12',
    };
    this.tensors = [];
    this.kernels = [];
    this.symbolTable = new Map();
    this.opCounter = 0;
  }

  generate() {
    // Extract tensors and kernels from AST
    for (const stmt of this.ast.statements) {
      if (stmt.type !== 'Glyph') continue;

      if (stmt.glyph === 'Wo') {
        this.extractTensor(stmt);
      } else if (stmt.glyph === 'Pop') {
        this.extractKernel(stmt);
      }
    }

    return {
      $schema: 'https://kuhul.dev/kson/v1',
      version: '1.0.0',
      manifest: this.manifest,
      tensors: this.tensors,
      kernels: this.kernels,
      schedule: this.generateSchedule(),
    };
  }

  extractTensor(stmt) {
    const shape = this.getArg(stmt, 'shape');
    const dtype = this.getArg(stmt, 'dtype') || 'float32';

    const shapeTuple = shape
      ? shape
          .split(',')
          .map(s => parseInt(s.trim()))
      : [];

    const tensor = {
      id: stmt.name,
      glyph: 'Wo',
      role: this.inferRole(stmt),
      shape: shapeTuple,
      dtype,
    };

    this.tensors.push(tensor);
    this.symbolTable.set(stmt.name, tensor);
  }

  extractKernel(stmt) {
    const kernel = {
      id: stmt.name,
      glyph: 'Pop',
      entry: `${stmt.name}_CS`,
      thread_group: [16, 16, 1],
      operations: this.extractOperations(stmt.body || []),
    };

    this.kernels.push(kernel);
  }

  extractOperations(body) {
    const ops = [];
    let phaseIndex = 0;
    const phases = ['load', 'π/4', 'π/2', '3π/4', 'π', 'store'];

    for (const stmt of body) {
      if (stmt.type !== 'Glyph') continue;

      const op = {
        id: `op${this.opCounter++}`,
        phase: phases[phaseIndex % phases.length],
        glyph: stmt.glyphValue,
        operation: this.glyphToOperation(stmt.glyph),
        inputs: this.getInputs(stmt),
        output: this.getOutput(stmt),
      };

      ops.push(op);
      phaseIndex++;
    }

    return ops;
  }

  glyphToOperation(glyph) {
    const mapping = {
      Yax: 'load',
      Chen: 'store',
      "Ch'en": 'store',
      Sek: '⊗',
      Xul: 'sync',
      '∇': 'gradient',
      '∫': 'integrate',
      '⊙': 'attend',
      '⊗': 'tensor_dot',
    };
    return mapping[glyph] || 'unknown';
  }

  getInputs(stmt) {
    const inputs = [];
    for (const arg of stmt.args) {
      if (arg.key && arg.key !== 'shape' && arg.key !== 'dtype') {
        inputs.push(arg.value);
      } else if (!arg.key && typeof arg.value === 'string') {
        inputs.push(arg.value);
      }
    }
    return inputs;
  }

  getOutput(stmt) {
    // Usually the named tensor
    return stmt.name || null;
  }

  inferRole(stmt) {
    // Heuristic: first tensor is input, last is output
    const index = this.tensors.length;
    const totalTensors = this.ast.statements.filter(s => s.type === 'Glyph' && s.glyph === 'WO').length;

    if (index === 0) return 'input';
    if (index === totalTensors - 1) return 'output';
    return 'scratch';
  }

  getArg(stmt, key) {
    const arg = stmt.args.find(a => a.key === key);
    return arg ? arg.value : null;
  }

  generateSchedule() {
    const kernel = this.kernels[0];
    if (!kernel) return {};

    return {
      dispatch: {
        kernel: kernel.id,
        grid: [64, 64, 1],
        phase_gate: 'π/2',
      },
    };
  }
}

// ============================================================================
// MAIN COMPILER
// ============================================================================

class KuhulCompiler {
  compile(sourceFile, outputFile = null, mode = 'validate') {
    try {
      // Read source
      if (!fs.existsSync(sourceFile)) {
        throw new Error(`Source file not found: ${sourceFile}`);
      }

      const source = fs.readFileSync(sourceFile, 'utf-8');
      console.log(`✓ Read source: ${sourceFile}`);

      // Phase 1: Tokenize
      const lexer = new KuhulLexer();
      let tokens = [];
      try {
        tokens = lexer.tokenize(source);
      } catch (err) {
        throw new Error(`Lexer error: ${err.toString()}`);
      }
      console.log(`✓ Lexer: ${tokens.length} tokens`);

      // Phase 2: Parse
      const parser = new KuhulParser(tokens);
      let ast = {};
      try {
        ast = parser.parse();
      } catch (err) {
        throw new Error(`Parser error: ${err.toString()}`);
      }
      console.log(`✓ Parser: ${ast.statements.length} statements`);

      // Phase 3: Semantic Analysis
      const analyzer = new KuhulSemanticAnalyzer(ast);
      try {
        analyzer.analyze();
      } catch (err) {
        throw new Error(`Semantic error: ${err.toString()}`);
      }
      console.log(`✓ Semantic analysis passed`);

      // Phase 4: Generate KSON
      const basename = path.basename(sourceFile, path.extname(sourceFile));
      const manifest = {
        name: basename,
        type: 'compute_kernel',
        target: 'directx_12',
      };

      const generator = new KSONGenerator(ast, manifest);
      const kson = generator.generate();
      console.log(`✓ KSON Generator: ${kson.tensors.length} tensors, ${kson.kernels.length} kernels`);

      // Validate KSON schema
      this.validateKSON(kson);
      console.log(`✓ KSON schema validation passed`);

      // Output
      if (outputFile) {
        const output = outputFile.endsWith('.kson') ? outputFile : `${outputFile}.kson`;
        fs.writeFileSync(output, JSON.stringify(kson, null, 2));
        console.log(`✓ Output: ${output}`);
      }

      console.log(`\n✅ Compilation successful`);
      return kson;
    } catch (err) {
      console.error(`\n❌ Compilation failed:`);
      console.error(`   ${err.message}`);
      process.exit(1);
    }
  }

  validateKSON(kson) {
    // Check schema
    if (!kson.$schema) throw new Error('KSON missing $schema');
    if (!kson.version) throw new Error('KSON missing version');
    if (!kson.manifest) throw new Error('KSON missing manifest');
    if (!Array.isArray(kson.tensors)) throw new Error('KSON tensors not an array');
    if (!Array.isArray(kson.kernels)) throw new Error('KSON kernels not an array');

    // Validate manifest
    if (!kson.manifest.name) throw new Error('manifest missing name');
    if (!kson.manifest.type) throw new Error('manifest missing type');
    if (!kson.manifest.target) throw new Error('manifest missing target');

    // Validate tensors
    for (const t of kson.tensors) {
      if (!t.id) throw new Error('tensor missing id');
      if (!t.glyph) throw new Error(`tensor ${t.id} missing glyph`);
      if (!t.role) throw new Error(`tensor ${t.id} missing role`);
      if (!Array.isArray(t.shape)) throw new Error(`tensor ${t.id} shape not an array`);
      if (!t.dtype) throw new Error(`tensor ${t.id} missing dtype`);
    }

    // Validate kernels
    for (const k of kson.kernels) {
      if (!k.id) throw new Error('kernel missing id');
      if (!k.glyph) throw new Error(`kernel ${k.id} missing glyph`);
      if (!k.entry) throw new Error(`kernel ${k.id} missing entry`);
      if (!Array.isArray(k.thread_group)) throw new Error(`kernel ${k.id} thread_group not an array`);
      if (!Array.isArray(k.operations)) throw new Error(`kernel ${k.id} operations not an array`);

      for (const op of k.operations) {
        if (!op.id) throw new Error(`operation missing id in kernel ${k.id}`);
        if (!op.phase) throw new Error(`operation ${op.id} missing phase in kernel ${k.id}`);
        if (!op.glyph) throw new Error(`operation ${op.id} missing glyph in kernel ${k.id}`);
        if (!op.operation) throw new Error(`operation ${op.id} missing operation name in kernel ${k.id}`);
      }
    }
  }
}

// ============================================================================
// CLI
// ============================================================================

if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log(`Usage: node kuhulc.js <source.kuhul> [output] [mode]`);
    console.log(`  source.kuhul: K'uhul source file`);
    console.log(`  output: Output file (default: source basename + .kson)`);
    console.log(`  mode: 'validate' (default) or other backend`);
    process.exit(1);
  }

  const sourceFile = args[0];
  const outputFile = args[1] || null;
  const mode = args[2] || 'validate';

  const compiler = new KuhulCompiler();
  compiler.compile(sourceFile, outputFile, mode);
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  KuhulLexer,
  KuhulParser,
  KuhulSemanticAnalyzer,
  KSONGenerator,
  KuhulCompiler,
  KuhulSyntaxError,
  SemanticError,
};
