// SPDX-License-Identifier: GPL-3.0-or-later
// Clunk — LLVM IR superoptimiser
// Copyright (C) 2025 Clunk contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

/*
 * Clunk IR Parser — reads LLVM IR text (.ll) into our in-memory representation.
 *
 * Handles a practical subset of LLVM IR sufficient for superoptimisation.
 * Unknown constructs are silently skipped.  Only genuinely ambiguous or
 * clearly malformed input produces a warning.
 */
#include "clunk/Parser/IRParser.h"

#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>
#include <algorithm>
#include <unordered_set>

namespace clunk::parser {

// ── Helper sets ────────────────────────────────────────────────────────────

static bool is_punct(char c) {
    return c == '(' || c == ')' || c == '{' || c == '}' || c == '[' || c == ']' ||
           c == '=' || c == ',' || c == ';' || c == ':' || c == '*' || c == '<' ||
           c == '>' || c == '!' || c == '@' || c == '%';
    // '#' and '.' are handled specially in the tokenizer
}

// Keywords that appear in LLVM IR and should be tokenized as Keyword type.
// Value keywords (undef, null, true, false, etc.) are left as Identifiers
// so that parse_value() can recognise them.
static const std::unordered_set<std::string> llvm_keywords = {
    // Top-level
    "define", "declare", "target", "source_filename", "attributes",
    "module", "asm", "type",
    // Linkage
    "private", "internal", "available_externally",
    "linkonce", "weak", "common", "appending", "extern_weak", "linkonce_odr",
    "weak_odr", "external",
    // Visibility / preemption
    "default", "hidden", "protected",
    "dso_local", "dso_preemptable",
    // Calling conventions
    "ccc", "fastcc", "coldcc", "webkit_jscc", "anyregcc",
    "preserve_mostcc", "preserve_allcc", "swiftcc", "swifttailcc",
    "cfguard_checkcc", "c",
    // Global kind
    "global", "constant",
    // Parameter attributes
    "signext", "zeroext", "byval", "sret", "noalias", "nocapture", "nest",
    "returned", "nonnull", "dereferenceable", "dereferenceable_or_null",
    "inreg", "swiftself", "swiftasync", "swift_error", "noundef",
    "elementtype", "align", "readonly", "writeonly", "readnone", "immarg",
    "allocptr", "byref", "preallocated", "fpclass", "nofpclass",
    // "captures(...)" (LLVM 18+) replaced the standalone "nocapture"
    // attribute with a parenthesised capture-component list, e.g.
    // `captures(none)`, `captures(address)`,
    // `captures(address, read_provenance)`. "writable", "dead_on_unwind"
    // and "initializes(...)" are newer memory-effect param attributes
    // clang emits alongside it. Without these in the keyword table, the
    // lexer tokenizes "captures" as a plain Identifier, so
    // collect_param_attrs' `while (peek == Keyword)` loop stops right
    // before it, and each of "captures" / "(" / "none" then gets fed to
    // parse_type() one at a time — which doesn't recognize any of them
    // and falls through to its "unknown token -> void" fallback, turning
    // one real parameter into a cascade of bogus void "parameters" and
    // consuming the WRONG closing paren as the end of the parameter
    // list (see the "ptr, void, void, void" class of bug reports).
    "captures", "writable", "dead_on_unwind", "initializes", "range",
    // Function attributes
    "nounwind", "uwtable", "mustprogress", "nosync",
    "nofree", "willreturn", "optnone", "noinline", "alwaysinline",
    "speculatable", "strictfp",
    "sanitize_address", "sanitize_memory", "sanitize_thread",
    "sanitize_hwaddress", "sanitize_memtag",
    "preemptable", "entry_count",
    // Binary-op / instruction flags
    "nuw", "nsw", "exact", "nneg", "disjoint",
    // Fast-math flags
    "fast", "nnel", "contract", "afn", "reassoc", "nsz", "arcp",
    // Memory operation modifiers
    "volatile", "inbounds", "atomic",
    // Atomic ordering
    "acquire", "release", "acq_rel", "seq_cst", "monotonic", "unordered",
    "syncscope",
    // Instruction keywords
    "label", "to",
    // Tail-call markers
    "tail", "notail", "musttail",
    // Global modifiers
    "section", "comdat", "thread_local", "localdynamic",
    "initialexec", "localexec", "unnamed_addr", "local_unnamed_addr",
    "externally_initialized",
    // Misc
    "max", "min", "cold", "hot",
    "sw", "uw", "splat", "from", "cleanup", "catch", "filter",
    "within", "uses", "alignstack", "allockind", "allocsize",
};

static const std::unordered_set<std::string> binop_names = {
    "add", "sub", "mul", "udiv", "sdiv", "urem", "srem",
    "fadd", "fsub", "fmul", "fdiv", "frem",
    "and", "or", "xor", "shl", "lshr", "ashr"
};

static const std::unordered_set<std::string> conversion_names = {
    "trunc", "zext", "sext", "fptrunc", "fpext",
    "fptoui", "fptosi", "uitofp", "sitofp",
    "ptrtoint", "inttoptr", "bitcast", "addrspacecast"
};

static const std::unordered_set<std::string> memop_names = {
    "alloca", "load", "store", "getelementptr",
    "fence", "cmpxchg", "atomicrmw"
};

static const std::unordered_set<std::string> terminator_names = {
    "ret", "br", "switch", "invoke", "resume", "unreachable"
};

static const std::unordered_set<std::string> other_inst_names = {
    "icmp", "fcmp", "phi", "call", "select",
    "extractvalue", "insertvalue",
    "extractelement", "insertelement", "shufflevector",
    "landingpad", "va_arg", "freeze"
};

static bool is_known_instruction(const std::string& name) {
    return binop_names.count(name) || conversion_names.count(name) ||
           memop_names.count(name) || terminator_names.count(name) ||
           other_inst_names.count(name);
}

// ── Constructor ────────────────────────────────────────────────────────────

IRParser::IRParser() = default;

// ── Public API ─────────────────────────────────────────────────────────────

std::shared_ptr<ir::Module> IRParser::parse_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw ParseError("Cannot open file: " + path);
    }
    std::stringstream buf;
    buf << ifs.rdbuf();
    return parse_string(buf.str());
}

std::shared_ptr<ir::Module> IRParser::parse_string(const std::string& ir_text) {
    auto mod = std::make_shared<ir::Module>();
    tokens_ = tokenize(ir_text);
    pos_ = 0;
    value_table_.clear();
    type_table_.clear();

    parse_module(mod);
    return mod;
}

void IRParser::register_type_resolver(TypeResolver resolver) {
    type_resolvers_.push_back(std::move(resolver));
}

// ── Lexer ──────────────────────────────────────────────────────────────────

std::vector<IRParser::Token> IRParser::tokenize(const std::string& source) {
    std::vector<Token> tokens;
    // Heuristic: average token length is ~4 characters (punctuation, short
    // identifiers, small numbers). Pre-reserving avoids repeated vector
    // re-allocations as tokens are appended.
    tokens.reserve(source.size() / 4);
    size_t i = 0;
    size_t line = 1;
    size_t col = 1;

    auto advance = [&]() {
        if (i < source.size()) {
            if (source[i] == '\n') { ++line; col = 1; }
            else { ++col; }
            ++i;
        }
    };

    auto peek_at = [&](size_t offset) -> char {
        size_t idx = i + offset;
        return idx < source.size() ? source[idx] : '\0';
    };

    while (i < source.size()) {
        char c = source[i];

        // Skip whitespace
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance();
            continue;
        }

        // Skip line comments
        if (c == ';') {
            while (i < source.size() && source[i] != '\n') advance();
            continue;
        }

        // String literal
        if (c == '"') {
            Token tok;
            tok.loc = {line, col};
            tok.type = TokenType::String;
            tok.text = "\"";
            tok.text.reserve(8);
            advance(); // opening quote
            while (i < source.size() && source[i] != '"') {
                if (source[i] == '\\' && i + 1 < source.size()) {
                    tok.text += source[i];
                    advance();
                }
                tok.text += source[i];
                advance();
            }
            if (i < source.size()) {
                tok.text += '"';
                advance(); // closing quote
            }
            tokens.push_back(tok);
            continue;
        }

        // Vararg ellipsis: ...
        if (c == '.' && peek_at(1) == '.' && peek_at(2) == '.') {
            Token tok;
            tok.loc = {line, col};
            tok.type = TokenType::Punctuation;
            tok.text = "...";
            advance(); advance(); advance();
            tokens.push_back(tok);
            continue;
        }

        // Attribute group reference: #N  (e.g. #0, #1)
        if (c == '#') {
            Token tok;
            tok.loc = {line, col};
            tok.type = TokenType::Punctuation;
            tok.text = "#";
            tok.text.reserve(8);
            advance();
            if (i < source.size() && std::isdigit(source[i])) {
                while (i < source.size() && std::isdigit(source[i])) {
                    tok.text += source[i];
                    advance();
                }
            }
            tokens.push_back(tok);
            continue;
        }

        // Punctuation
        if (is_punct(c)) {
            Token tok;
            tok.loc = {line, col};
            tok.type = TokenType::Punctuation;
            tok.text = std::string(1, c);
            advance();
            tokens.push_back(tok);
            continue;
        }

        // Numbers (including negative, hex, float)
        if (std::isdigit(c) || (c == '-' && i + 1 < source.size() &&
                                 (std::isdigit(source[i + 1]) || source[i + 1] == '.'))) {
            bool is_neg = (c == '-');
            Token tok;
            tok.loc = {line, col};
            tok.type = TokenType::Number;
            tok.text.reserve(16);  // numbers can be longer (hex, floats)
            if (is_neg) {
                tok.text = "-";
                advance();
            }
            // Integer part
            while (i < source.size() && std::isdigit(source[i])) {
                tok.text += source[i];
                advance();
            }
            // Hex (0x...)
            if (i < source.size() && source[i] == 'x' && tok.text == "0") {
                tok.text += 'x';
                advance();
                while (i < source.size() && std::isxdigit(source[i])) {
                    tok.text += source[i];
                    advance();
                }
            }
            // Floating point — make sure the '.' is not the start of '...'
            if (i < source.size() && source[i] == '.' &&
                !(i + 2 < source.size() && source[i + 1] == '.' && source[i + 2] == '.')) {
                tok.text += '.';
                advance();
                while (i < source.size() && std::isdigit(source[i])) {
                    tok.text += source[i];
                    advance();
                }
                // Exponent
                if (i < source.size() && (source[i] == 'e' || source[i] == 'E')) {
                    tok.text += source[i];
                    advance();
                    if (i < source.size() && (source[i] == '+' || source[i] == '-')) {
                        tok.text += source[i];
                        advance();
                    }
                    while (i < source.size() && std::isdigit(source[i])) {
                        tok.text += source[i];
                        advance();
                    }
                }
            }
            tokens.push_back(tok);
            continue;
        }

        // Identifiers and keywords
        if (std::isalpha(c) || c == '_' || c == '.' || c == '$') {
            Token tok;
            tok.loc = {line, col};
            tok.text.clear();
            tok.text.reserve(16);  // identifiers vary in length; reserve a small buffer
            while (i < source.size() && (std::isalnum(source[i]) || source[i] == '_' ||
                                          source[i] == '.' || source[i] == '$')) {
                tok.text += source[i];
                advance();
            }

            // Classify: type token, keyword, or plain identifier?
            if (tok.text.size() > 1 && tok.text[0] == 'i' &&
                std::all_of(tok.text.begin() + 1, tok.text.end(), ::isdigit)) {
                tok.type = TokenType::Type;
            } else if (tok.text == "void" || tok.text == "float" || tok.text == "double" ||
                       tok.text == "half" || tok.text == "bfloat" ||
                       tok.text == "x86_fp80" || tok.text == "fp128" || tok.text == "ppc_fp128" ||
                       tok.text == "ptr" || tok.text == "token" || tok.text == "opaque") {
                tok.type = TokenType::Type;
            } else if (llvm_keywords.count(tok.text)) {
                tok.type = TokenType::Keyword;
            } else {
                tok.type = TokenType::Identifier;
            }
            tokens.push_back(tok);
            continue;
        }

        // Skip unknown characters (e.g. '-', '|', '~', etc.)
        advance();
    }

    // EOF token
    Token eof;
    eof.type = TokenType::Eof;
    eof.text = "";
    eof.loc = {line, col};
    tokens.push_back(eof);

    return tokens;
}

// ── Token access ───────────────────────────────────────────────────────────

IRParser::Token IRParser::next_token() {
    if (pos_ < tokens_.size()) {
        return tokens_[pos_++];
    }
    return tokens_.back(); // EOF
}

IRParser::Token IRParser::peek_token() const {
    if (pos_ < tokens_.size()) {
        return tokens_[pos_];
    }
    return tokens_.back(); // EOF
}

bool IRParser::match(TokenType type, const std::string& text) {
    auto tok = peek_token();
    if (tok.type == type && (text.empty() || tok.text == text)) {
        pos_++;
        return true;
    }
    return false;
}

IRParser::Token IRParser::expect(TokenType type, const std::string& text) {
    auto tok = next_token();
    if (tok.type != type) {
        throw ParseError("Expected token type " + std::to_string(static_cast<int>(type)) +
                         " but got '" + tok.text + "' at line " +
                         std::to_string(tok.loc.line) + " col " +
                         std::to_string(tok.loc.col));
    }
    if (!text.empty() && tok.text != text) {
        throw ParseError("Expected '" + text + "' but got '" + tok.text +
                         "' at line " + std::to_string(tok.loc.line) +
                         " col " + std::to_string(tok.loc.col));
    }
    return tok;
}

// ── Skip helpers ───────────────────────────────────────────────────────────

void IRParser::skip_balanced(char open, char close) {
    // Punctuation tokens are always single-character strings, so we compare
    // against the char directly to avoid constructing std::string temporaries
    // on every iteration of the loop below.
    auto is_punct_char = [](const Token& tok, char ch) {
        return tok.type == TokenType::Punctuation &&
               tok.text.size() == 1 && tok.text[0] == ch;
    };

    // Consume the opening token
    if (!is_punct_char(peek_token(), open)) {
        return;
    }
    next_token(); // consume open

    int depth = 1;
    while (peek_token().type != TokenType::Eof && depth > 0) {
        auto tok = next_token();
        if (is_punct_char(tok, open))
            depth++;
        else if (is_punct_char(tok, close))
            depth--;
    }
}

void IRParser::skip_metadata() {
    // Consume leading '!'
    if (!(peek_token().type == TokenType::Punctuation && peek_token().text == "!"))
        return;
    next_token(); // consume '!'

    auto tok = peek_token();
    if (tok.type == TokenType::Number) {
        // !N — numeric metadata reference
        next_token();
    } else if (tok.type == TokenType::Identifier) {
        // !name — named metadata kind (e.g. !dbg, !range, !tbaa)
        next_token();
        // May be followed by !N, !{...}, or !name
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "!") {
            next_token(); // consume '!'
            tok = peek_token();
            if (tok.type == TokenType::Number) {
                next_token();
            } else if (tok.type == TokenType::Punctuation && tok.text == "{") {
                skip_balanced('{', '}');
            } else if (tok.type == TokenType::Identifier) {
                next_token(); // consume metadata kind name
                // May have further ! ref
                if (peek_token().type == TokenType::Punctuation && peek_token().text == "!") {
                    next_token();
                    if (peek_token().type == TokenType::Number) next_token();
                    else if (peek_token().type == TokenType::Punctuation &&
                             peek_token().text == "{") {
                        skip_balanced('{', '}');
                    }
                }
            } else if (tok.type == TokenType::Punctuation && tok.text == "(") {
                skip_balanced('(', ')');
            }
        } else if (peek_token().type == TokenType::Punctuation && peek_token().text == "{") {
            // !name { ... }
            skip_balanced('{', '}');
        }
    } else if (tok.type == TokenType::Punctuation && tok.text == "{") {
        // !{ ... } — metadata tuple
        skip_balanced('{', '}');
    } else if (tok.type == TokenType::String) {
        // !"string"
        next_token();
    }
    // else: nothing recognizable after '!', just return
}

void IRParser::skip_trailing_metadata() {
    // Consume [, !kind !val]* patterns that appear after instructions
    while (true) {
        // Check for comma leading to metadata
        if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
            // Peek ahead — is the next token after comma a '!'?
            if (pos_ + 1 < tokens_.size() &&
                tokens_[pos_ + 1].type == TokenType::Punctuation &&
                tokens_[pos_ + 1].text == "!") {
                next_token(); // consume ','
            } else {
                break; // comma is not for metadata
            }
        }
        // Check for '!'
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "!") {
            skip_metadata();
        } else {
            break;
        }
    }
}

void IRParser::skip_param_attrs() {
    // Skip all parameter attributes that can appear before a type.
    // This includes keywords with optional parenthesized args like
    // dereferenceable(N), align(N), elementtype(...), byval(T), etc.
    while (true) {
        auto tok = peek_token();
        if (tok.type == TokenType::Keyword) {
            std::string kw = tok.text;
            next_token(); // consume the keyword
            // If followed by '(' consume the parenthesized argument
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                skip_balanced('(', ')');
            }
            // 'align N' — only 'align' takes a bare number argument
            if (kw == "align" && peek_token().type == TokenType::Number) {
                next_token(); // consume alignment count
            }
        } else {
            break;
        }
    }
}

void IRParser::skip_function_attrs_until_brace() {
    // Skip all tokens between ')' and '{' in a function definition.
    // This handles #N, nounwind, uwtable, "key"="val", etc.
    while (peek_token().type != TokenType::Eof) {
        auto tok = peek_token();
        if (tok.type == TokenType::Punctuation && tok.text == "{") break;
        next_token();
    }
}

// ── Top-level parsing ──────────────────────────────────────────────────────

void IRParser::parse_module(std::shared_ptr<ir::Module> mod) {
    while (peek_token().type != TokenType::Eof) {
        auto tok = peek_token();

        // Metadata attachments and definitions: !N = ..., !{...}
        if (tok.type == TokenType::Punctuation && tok.text == "!") {
            // Look ahead: if this is !llvm.module.flags, parse it specially
            if (pos_ + 1 < tokens_.size() &&
                tokens_[pos_ + 1].type == TokenType::Identifier &&
                tokens_[pos_ + 1].text == "llvm.module.flags") {
                parse_llvm_module_flags(mod);
                continue;
            }
            // Named metadata definition: !<name> = !{ ... }
            // (e.g. !llvm.ident, !opencl.ocl.version, !dbg.cu, !llvm.dbg.cu)
            if (pos_ + 1 < tokens_.size() &&
                tokens_[pos_ + 1].type == TokenType::Identifier &&
                pos_ + 2 < tokens_.size() &&
                tokens_[pos_ + 2].type == TokenType::Punctuation &&
                tokens_[pos_ + 2].text == "=") {
                parse_named_metadata(mod);
                continue;
            }
            // Check for !N = !{ i32 <behavior>, !"key", ... } (module flag entry)
            if (pos_ + 1 < tokens_.size() &&
                tokens_[pos_ + 1].type == TokenType::Number) {
                if (try_parse_module_flag_entry(mod)) {
                    continue;
                }
                // Not a module-flag entry — but might be a plain !N = !{...}
                // metadata definition. parse_metadata_def will detect and
                // consume it; if it doesn't match either form, fall through.
                if (pos_ + 2 < tokens_.size() &&
                    tokens_[pos_ + 2].type == TokenType::Punctuation &&
                    tokens_[pos_ + 2].text == "=") {
                    parse_metadata_def(mod);
                    continue;
                }
            }
            // Bare metadata attachment (no '=') — silently consume
            skip_metadata();
            // If there's an '=' after, skip the rest until the next top-level construct
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "=") {
                next_token(); // consume '='
                while (peek_token().type != TokenType::Eof) {
                    auto next = peek_token();
                    if (next.type == TokenType::Keyword &&
                        (next.text == "define" || next.text == "declare" ||
                         next.text == "target" || next.text == "attributes" ||
                         next.text == "module" || next.text == "source_filename")) break;
                    if (next.type == TokenType::Punctuation && next.text == "@") break;
                    if (next.type == TokenType::Punctuation && next.text == "%") {
                        // Could be a type definition
                        if (pos_ + 2 < tokens_.size() &&
                            tokens_[pos_ + 1].type == TokenType::Identifier &&
                            tokens_[pos_ + 2].type == TokenType::Punctuation &&
                            tokens_[pos_ + 2].text == "=") break;
                    }
                    if (next.type == TokenType::Punctuation && next.text == "!") {
                        // Next metadata definition
                        break;
                    }
                    next_token();
                }
            }
            continue;
        }

        // target triple = "..." / target datalayout = "..."
        if (tok.type == TokenType::Keyword && tok.text == "target") {
            parse_target(mod);
            continue;
        }

        // source_filename = "..."
        // Note: `source_filename` is in llvm_keywords, so the lexer
        // classifies it as a Keyword — not an Identifier.
        if (tok.type == TokenType::Keyword && tok.text == "source_filename") {
            next_token(); // consume 'source_filename'
            match(TokenType::Punctuation, "=");
            if (peek_token().type == TokenType::String) {
                std::string sf = peek_token().text;
                if (sf.size() >= 2 && sf.front() == '"' && sf.back() == '"') {
                    sf = sf.substr(1, sf.size() - 2);
                }
                mod->set_source_filename(sf);
            }
            next_token(); // consume the string value
            continue;
        }

        // Named type: %struct.X = type { ... }
        if (tok.type == TokenType::Punctuation && tok.text == "%") {
            if (pos_ + 3 < tokens_.size() &&
                tokens_[pos_ + 1].type == TokenType::Identifier &&
                tokens_[pos_ + 2].type == TokenType::Punctuation &&
                tokens_[pos_ + 2].text == "=" &&
                (tokens_[pos_ + 3].type == TokenType::Identifier &&
                 tokens_[pos_ + 3].text == "type")) {
                next_token(); // consume '%'
                auto name_tok = next_token(); // type name
                next_token(); // consume '='
                next_token(); // consume 'type'
                auto ty = parse_type();
                type_table_[name_tok.text] = ty;
                mod->add_named_type(name_tok.text, ty);
                continue;
            }
        }

        // Function definition: define ...
        if (tok.type == TokenType::Keyword && tok.text == "define") {
            parse_function_def(mod);
            continue;
        }

        // Function declaration: declare ...
        if (tok.type == TokenType::Keyword && tok.text == "declare") {
            parse_function_decl(mod);
            continue;
        }

        // Global variable: @name = ...
        if (tok.type == TokenType::Punctuation && tok.text == "@") {
            parse_global(mod);
            continue;
        }

        // Attributes group: attributes #N = { ... }
        // Capture verbatim so function references like `define ... #0 { }`
        // still resolve after round-trip.
        if (tok.type == TokenType::Keyword && tok.text == "attributes") {
            parse_attribute_group(mod);
            continue;
        }

        // module asm "..."
        if (tok.type == TokenType::Keyword && tok.text == "module") {
            parse_module_asm(mod);
            continue;
        }

        // Unknown top-level construct — skip silently (no warning)
        next_token();
    }
}

void IRParser::parse_target(std::shared_ptr<ir::Module> mod) {
    next_token(); // consume 'target'
    auto what = next_token(); // 'triple' or 'datalayout'
    match(TokenType::Punctuation, "=");
    auto val = next_token(); // the string value

    ir::TargetInfo info = mod->target();
    if (what.text == "triple") {
        info.triple = val.text;
        if (info.triple.size() >= 2 && info.triple.front() == '"' && info.triple.back() == '"') {
            info.triple = info.triple.substr(1, info.triple.size() - 2);
        }
    } else if (what.text == "datalayout") {
        info.datalayout = val.text;
        if (info.datalayout.size() >= 2 && info.datalayout.front() == '"' && info.datalayout.back() == '"') {
            info.datalayout = info.datalayout.substr(1, info.datalayout.size() - 2);
        }
    }
    mod->set_target(info);
}

void IRParser::parse_global(std::shared_ptr<ir::Module> mod) {
    // @name = [linkage] [constant] type value
    next_token(); // consume '@'
    auto name_tok = next_token(); // global name
    match(TokenType::Punctuation, "=");

    ir::GlobalValue gv;
    gv.name = "@" + name_tok.text;

    // Parse optional linkage / global / constant keywords. Unlike the
    // previous version of this loop (which threw every keyword but
    // "constant"/"global" away), we now record the actual linkage so it
    // round-trips, and remember whether we saw a linkage that implies
    // "no initializer follows" (external / extern_weak) — LLVM requires
    // one of those keywords whenever a global has no initializer value,
    // and the printer needs to know to re-emit it (see is_declaration).
    while (peek_token().type == TokenType::Keyword) {
        auto kw = peek_token().text;
        if (kw == "constant") { gv.is_constant = true; next_token(); break; }
        if (kw == "global")   { next_token(); break; }
        if (kw == "internal")    { gv.linkage = ir::Linkage::Internal;    next_token(); continue; }
        if (kw == "private")     { gv.linkage = ir::Linkage::Private;     next_token(); continue; }
        if (kw == "weak")        { gv.linkage = ir::Linkage::Weak;        next_token(); continue; }
        if (kw == "linkonce")    { gv.linkage = ir::Linkage::LinkOnce;    next_token(); continue; }
        if (kw == "linkonce_odr"){ gv.linkage = ir::Linkage::LinkOnceODR; next_token(); continue; }
        if (kw == "weak_odr")    { gv.linkage = ir::Linkage::WeakODR;     next_token(); continue; }
        if (kw == "common")      { gv.linkage = ir::Linkage::Common;      next_token(); continue; }
        if (kw == "appending")   { gv.linkage = ir::Linkage::Appending;   next_token(); continue; }
        if (kw == "available_externally") {
            gv.linkage = ir::Linkage::AvailableExternally; next_token(); continue;
        }
        if (kw == "external") {
            gv.linkage = ir::Linkage::External;
            gv.is_declaration = true;
            next_token();
            continue;
        }
        if (kw == "extern_weak") {
            gv.linkage = ir::Linkage::ExternalWeak;
            gv.is_declaration = true;
            next_token();
            continue;
        }
        // Skip other keywords (dso_local, unnamed_addr, thread_local, etc.)
        next_token();
    }

    // Try to parse the type
    if (peek_token().type == TokenType::Type ||
        (peek_token().type == TokenType::Punctuation && peek_token().text == "%") ||
        (peek_token().type == TokenType::Punctuation && peek_token().text == "[")) {
        gv.type = parse_type();
    }

    // Capture the initialiser (and any trailing attrs) as a raw string so
    // we can re-emit it verbatim. We rebuild it from tokens until we hit
    // the next top-level entity (define/declare/@name/%name=).
    {
        std::string init_str;
        bool first = true;
        while (peek_token().type != TokenType::Eof) {
            auto t = peek_token();
            if (t.type == TokenType::Keyword &&
                (t.text == "define" || t.text == "declare" || t.text == "target" ||
                 t.text == "attributes" || t.text == "module")) break;
            if (t.type == TokenType::Punctuation && t.text == "@") break;
            if (t.type == TokenType::Punctuation && t.text == "%") {
                if (pos_ + 2 < tokens_.size() &&
                    tokens_[pos_ + 1].type == TokenType::Identifier &&
                    tokens_[pos_ + 2].type == TokenType::Punctuation &&
                    tokens_[pos_ + 2].text == "=") {
                    break;
                }
            }
            if (t.type == TokenType::Punctuation && t.text == "!") break;
            // Append the token text. Use a space separator unless the
            // current/previous token is a structural punctuation that
            // doesn't want a space (e.g. '[', ']', 'c', '"').
            if (!first) {
                bool no_space = (t.text == "," || t.text == "]" || t.text == ")" ||
                                 t.text == "x" ||
                                 (init_str.back() == '[' || init_str.back() == ' ' ||
                                  init_str.back() == 'c' || init_str.back() == '"'));
                init_str += no_space ? "" : " ";
            }
            init_str += t.text;
            first = false;
            next_token();
        }
        gv.init_value = init_str;

        // Defensive fallback: even if we didn't see an explicit
        // `external`/`extern_weak` keyword (e.g. a hand-written/odd input
        // that omits it), a captured string that is empty or starts
        // directly with "," means no initializer VALUE token was present
        // before the trailing attributes (align/section/...) — i.e. this
        // is structurally a declaration. Without this, Module::to_string()
        // would print `global <type> , align N` (a leading-comma-after-
        // space typo LLVM rejects with "expected value token") instead of
        // `external global <type>, align N`.
        if (!gv.is_declaration &&
            (gv.init_value.empty() || gv.init_value.front() == ',')) {
            gv.is_declaration = true;
        }
    }

    mod->add_global(gv);
}

void IRParser::parse_function_decl(std::shared_ptr<ir::Module> mod) {
    next_token(); // consume 'declare'

    // Skip linkage, cconv, ret attrs
    while (peek_token().type == TokenType::Keyword) {
        auto kw = peek_token().text;
        // Stop before a type token — the next keyword might be a return-type
        // attribute we should skip, but we need to be careful not to consume
        // the type itself.
        if (kw == "dso_local" || kw == "nounwind" || kw == "readnone" ||
            kw == "readonly" || kw == "writeonly" || kw == "willreturn" ||
            kw == "nofree" || kw == "nosync" || kw == "optnone" ||
            kw == "alwaysinline" || kw == "noinline" || kw == "internal" ||
            kw == "private" || kw == "external" || kw == "weak" ||
            kw == "linkonce" || kw == "linkonce_odr" || kw == "weak_odr" ||
            kw == "hidden" || kw == "protected" || kw == "default" ||
            kw == "available_externally" || kw == "extern_weak" ||
            kw == "mustprogress" || kw == "strictfp" ||
            kw == "sanitize_address" || kw == "sanitize_memory" ||
            kw == "sanitize_thread" || kw == "speculatable" ||
            kw == "zeroext" || kw == "signext" || kw == "uwtable" ||
            kw == "coldcc" || kw == "fastcc" || kw == "ccc") {
            next_token();
            // Handle parenthesized args
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                skip_balanced('(', ')');
            }
            continue;
        }
        // Unknown keyword before the return type — could be a return attr.
        // Skip it but be cautious.
        next_token();
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
            skip_balanced('(', ')');
        }
    }

    // Return type
    auto ret_type = parse_type();

    // Function name: @name
    std::string fn_name = parse_name();

    // Arguments: ( ... )
    expect(TokenType::Punctuation, "(");
    std::vector<std::shared_ptr<ir::Type>> param_types;
    // Track per-argument parameter attributes so we can re-emit them on
    // declarations.  Previously declarations dropped parameter attributes
    // like `noundef`, `noalias`, `readonly` — which clang then warned
    // about as a signature mismatch on link.
    struct DeclArg {
        std::shared_ptr<ir::Type> type;
        std::unordered_map<std::string, std::string> attrs;
    };
    std::vector<DeclArg> decl_args;

    auto collect_decl_param_attrs = [this](std::unordered_map<std::string, std::string>& attrs) {
        while (peek_token().type == TokenType::Keyword) {
            std::string kw = peek_token().text;
            next_token();
            // Record the aliasing-relevant attrs the memory optimiser
            // cares about; everything else is silently consumed.
            if (kw == "noalias" || kw == "readonly" || kw == "readnone" ||
                kw == "nocapture" || kw == "noundef" || kw == "nonnull" ||
                kw == "writeonly" || kw == "noalias" || kw == "returned" ||
                kw == "inreg" || kw == "signext" || kw == "zeroext") {
                attrs[kw] = "true";
            }
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                skip_balanced('(', ')');
            }
            if (kw == "align" && peek_token().type == TokenType::Number) {
                attrs["align"] = peek_token().text;
                next_token();
            }
            if (kw == "dereferenceable" && peek_token().type == TokenType::Punctuation &&
                peek_token().text == "(") {
                skip_balanced('(', ')');
            }
        }
    };

    while (!(peek_token().type == TokenType::Punctuation && peek_token().text == ")")) {
        DeclArg arg;
        // Parameter attributes before the type (noundef, noalias, etc.)
        collect_decl_param_attrs(arg.attrs);

        // Vararg: ...
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "...") {
            next_token();
            if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
                next_token();
            }
            continue;
        }

        arg.type = parse_type();
        param_types.push_back(arg.type);
        // Attributes can also appear between type and name
        collect_decl_param_attrs(arg.attrs);
        // Skip parameter name if present (%name) — declarations usually
        // don't name their parameters, but it's legal.
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "%") {
            next_token(); // '%'
            next_token(); // name
        }
        decl_args.push_back(std::move(arg));

        if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
            next_token();
        }
    }
    expect(TokenType::Punctuation, ")");

    // Collect trailing function attributes (#0, nounwind, uwtable, etc.)
    // so they round-trip.  Previously these were silently dropped, which
    // meant a `declare void @foo() #0` re-emitted as `declare void @foo()`
    // — losing the attribute-group reference and breaking clang recompilation.
    auto func_attrs = collect_function_attrs();

    auto fn_type = std::make_shared<ir::FunctionType>(ret_type, param_types);
    auto fn = std::make_shared<ir::Function>(fn_name, fn_type);
    for (const auto& a : func_attrs) fn->add_function_attribute(a);
    // Add arguments to the function so the emitter can render them.
    // Without this, `declare void @foo(i32 noundef)` was re-emitted as
    // `declare void @foo()` — losing the parameter list and breaking
    // clang's signature-mismatch check at link time.
    for (const auto& arg : decl_args) {
        fn->add_argument(arg.type, "", arg.attrs);
    }
    mod->add_function(fn);
}

void IRParser::parse_function_def(std::shared_ptr<ir::Module> mod) {
    next_token(); // consume 'define'

    // Skip linkage, cconv, ret attrs
    ir::Linkage linkage = ir::Linkage::External;
    while (peek_token().type == TokenType::Keyword) {
        auto kw = peek_token().text;
        if (kw == "internal")              { linkage = ir::Linkage::Internal; next_token(); continue; }
        if (kw == "private")               { linkage = ir::Linkage::Private;  next_token(); continue; }
        if (kw == "weak")                  { linkage = ir::Linkage::Weak;     next_token(); continue; }
        if (kw == "linkonce")              { linkage = ir::Linkage::LinkOnce; next_token(); continue; }
        if (kw == "linkonce_odr")          { linkage = ir::Linkage::LinkOnceODR; next_token(); continue; }
        if (kw == "weak_odr")              { linkage = ir::Linkage::WeakODR;  next_token(); continue; }
        if (kw == "external")              { linkage = ir::Linkage::External; next_token(); continue; }
        if (kw == "available_externally" ||
            kw == "extern_weak" ||
            kw == "appending" ||
            kw == "common" ||
            kw == "dso_local" || kw == "dso_preemptable" ||
            kw == "hidden" || kw == "protected" || kw == "default" ||
            kw == "nounwind" || kw == "readnone" || kw == "readonly" ||
            kw == "writeonly" || kw == "willreturn" || kw == "nofree" ||
            kw == "nosync" || kw == "optnone" || kw == "alwaysinline" ||
            kw == "noinline" || kw == "mustprogress" || kw == "strictfp" ||
            kw == "sanitize_address" || kw == "sanitize_memory" ||
            kw == "sanitize_thread" || kw == "speculatable" ||
            kw == "zeroext" || kw == "signext" || kw == "uwtable" ||
            kw == "coldcc" || kw == "fastcc" || kw == "ccc") {
            next_token();
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                skip_balanced('(', ')');
            }
            continue;
        }
        // Unknown keyword before the return type — skip it as a return attribute
        next_token();
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
            skip_balanced('(', ')');
        }
    }

    // Return type
    auto ret_type = parse_type();

    // Function name: @name
    std::string fn_name = parse_name();

    // Clear value table for this function
    value_table_.clear();

    // Arguments: ( ... )
    expect(TokenType::Punctuation, "(");
    std::vector<std::shared_ptr<ir::Type>> param_types;
    struct ParsedArg {
        std::shared_ptr<ir::Type> type;
        std::string name;
        std::unordered_map<std::string, std::string> attrs;
    };
    std::vector<ParsedArg> args;
    // Like skip_param_attrs(), but records the aliasing-relevant attrs the
    // memory optimiser consumes (noalias / readonly / readnone).
    auto collect_param_attrs = [this](std::unordered_map<std::string, std::string>& attrs) {
        while (peek_token().type == TokenType::Keyword) {
            std::string kw = peek_token().text;
            next_token();
            if (kw == "noalias" || kw == "readonly" || kw == "readnone" ||
                kw == "nocapture") {
                attrs[kw] = "true";
            }
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                skip_balanced('(', ')');
            }
            if (kw == "align" && peek_token().type == TokenType::Number) {
                attrs["align"] = peek_token().text;  // consumed by AlignOpt
                next_token();
            }
        }
    };
    while (!(peek_token().type == TokenType::Punctuation && peek_token().text == ")")) {
        ParsedArg arg;
        // Parameter attributes (recorded, not just skipped)
        collect_param_attrs(arg.attrs);

        // Vararg: ...
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "...") {
            next_token();
            if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
                next_token();
            }
            continue;
        }

        arg.type = parse_type();
        param_types.push_back(arg.type);

        // Attributes can also appear between type and name
        collect_param_attrs(arg.attrs);

        // Optional parameter name: %name
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "%") {
            arg.name = parse_name();
        }
        args.push_back(std::move(arg));

        if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
            next_token();
        }
    }
    expect(TokenType::Punctuation, ")");

    // Skip function attributes after params: #0, nounwind, uwtable, etc.
    // Just consume everything until '{'.
    // Collect function attributes after params: #0, nounwind, uwtable, etc.
    auto func_attrs = collect_function_attrs();

    // Create the function
    auto fn_type = std::make_shared<ir::FunctionType>(ret_type, param_types);
    auto fn = std::make_shared<ir::Function>(fn_name, fn_type, linkage);
    for (const auto& a : func_attrs) fn->add_function_attribute(a);

    // Add arguments and register them in value_table_
    for (size_t i = 0; i < args.size(); ++i) {
        fn->add_argument(args[i].type, args[i].name, args[i].attrs);
        if (!args[i].name.empty()) {
            auto arg_val = std::make_shared<ir::Value>(args[i].type, args[i].name);
            value_table_[args[i].name] = arg_val;
        }
    }

    // Parse function body: { basic_blocks }
    expect(TokenType::Punctuation, "{");

    ir::BasicBlock* current_bb = nullptr;

    while (!(peek_token().type == TokenType::Punctuation && peek_token().text == "}")) {
        if (peek_token().type == TokenType::Eof) {
            break;
        }

        // Skip metadata-only lines inside function body
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "!") {
            skip_trailing_metadata();
            continue;
        }

        // Check for basic block label: identifier followed by ':'
        if (peek_token().type == TokenType::Identifier &&
            pos_ + 1 < tokens_.size() &&
            tokens_[pos_ + 1].type == TokenType::Punctuation &&
            tokens_[pos_ + 1].text == ":") {
            auto label = next_token().text;
            next_token(); // consume ':'
            current_bb = &fn->add_block(label);
            continue;
        }

        // First block might not have a label — create an implicit "entry"
        if (current_bb == nullptr) {
            current_bb = &fn->add_block("entry");
        }

        // Try to parse an instruction
        try {
            auto inst = parse_instruction(*fn);
            if (inst) {
                current_bb->add_instruction(inst);

                // Register result value in value table
                if (inst->has_name()) {
                    value_table_[inst->name()] = inst;
                }
            }
        } catch (const ParseError& e) {
            // Skip to next recognizable point
            while (peek_token().type != TokenType::Eof &&
                   !(peek_token().type == TokenType::Punctuation && peek_token().text == "}") &&
                   !(peek_token().type == TokenType::Identifier &&
                     pos_ + 1 < tokens_.size() &&
                     tokens_[pos_ + 1].type == TokenType::Punctuation &&
                     tokens_[pos_ + 1].text == ":")) {
                next_token();
            }
        }
    }

    expect(TokenType::Punctuation, "}");

    // ── Forward-reference repair ─────────────────────────────────────────
    // A use BEFORE the definition (a loop phi's back-edge value, a phi
    // whose incoming comes from a later block) made parse_value create a
    // bare placeholder Value; when the real instruction was parsed later
    // it replaced the value_table_ entry, but every operand already
    // holding the placeholder kept it. Consumers that resolve values by
    // OBJECT identity (the Interpreter keys bindings by pointer) then
    // failed on every loop-carried phi. Rewire such operands to the
    // defining instruction.
    {
        std::unordered_map<std::string, std::shared_ptr<ir::Value>> defs;
        for (auto& block : fn->blocks()) {
            for (auto& inst : block->instructions()) {
                if (inst && inst->has_name()) defs[inst->name()] = inst;
            }
        }
        for (auto& block : fn->blocks()) {
            for (auto& inst : block->instructions()) {
                if (!inst) continue;
                for (size_t i = 0; i < inst->num_operands(); ++i) {
                    auto op = inst->operand(i);
                    if (!op || !op->has_name()) continue;
                    auto it = defs.find(op->name());
                    if (it != defs.end() && it->second != op) {
                        inst->set_operand(i, it->second);
                    }
                }
            }
        }
    }

    // Compute predecessor info
    fn->compute_predecessors();

    mod->add_function(fn);
}

// ── Type parsing ───────────────────────────────────────────────────────────

std::shared_ptr<ir::Type> IRParser::parse_type() {
    auto tok = peek_token();

    // Void
    if (tok.type == TokenType::Type && tok.text == "void") {
        next_token();
        return type_ctx_.void_type();
    }

    // Float
    if (tok.type == TokenType::Type && tok.text == "float") {
        next_token();
        return type_ctx_.float_type();
    }

    // Double
    if (tok.type == TokenType::Type && tok.text == "double") {
        next_token();
        return type_ctx_.double_type();
    }

    // Opaque pointer type: ptr (LLVM 15+)
    if (tok.type == TokenType::Type && tok.text == "ptr") {
        next_token();
        auto ty = type_ctx_.pointer_to(type_ctx_.void_type());
        // ptr addrspace(N)
        if (peek_token().type == TokenType::Keyword && peek_token().text == "addrspace") {
            next_token();
            skip_balanced('(', ')');
        }
        return ty;
    }

    // Token type
    if (tok.type == TokenType::Type && tok.text == "token") {
        next_token();
        return type_ctx_.void_type(); // approximate
    }

    // Opaque struct type
    if (tok.type == TokenType::Type && tok.text == "opaque") {
        next_token();
        std::shared_ptr<ir::Type> ty = std::make_shared<ir::StructType>(std::vector<std::shared_ptr<ir::Type>>{}, false);
        while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
            next_token();
            ty = type_ctx_.pointer_to(ty);
        }
        return ty;
    }

    // Half / bfloat
    if (tok.type == TokenType::Type && (tok.text == "half" || tok.text == "bfloat")) {
        next_token();
        std::shared_ptr<ir::Type> ty = type_ctx_.float_type(); // approximate
        while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
            next_token();
            ty = type_ctx_.pointer_to(ty);
        }
        return ty;
    }

    // x86_fp80, fp128, ppc_fp128
    if (tok.type == TokenType::Type && (tok.text == "x86_fp80" || tok.text == "fp128" || tok.text == "ppc_fp128")) {
        next_token();
        std::shared_ptr<ir::Type> ty = type_ctx_.double_type(); // approximate
        while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
            next_token();
            ty = type_ctx_.pointer_to(ty);
        }
        return ty;
    }

    // Integer: i1, i8, i32, i64, etc.
    if (tok.type == TokenType::Type && tok.text[0] == 'i') {
        next_token();
        unsigned bits = std::stoul(tok.text.substr(1));
        std::shared_ptr<ir::Type> ty = type_ctx_.get_int(bits);

        // Check for function-type suffix: i32 (i32, i32)*
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
            // Explicit function type: ret_type (param_types...)*
            next_token(); // consume '('
            std::vector<std::shared_ptr<ir::Type>> fparam_types;
            while (!(peek_token().type == TokenType::Punctuation && peek_token().text == ")")) {
                skip_param_attrs();
                if (peek_token().type == TokenType::Punctuation && peek_token().text == "...") {
                    next_token();
                    if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") next_token();
                    continue;
                }
                fparam_types.push_back(parse_type());
                if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") next_token();
            }
            expect(TokenType::Punctuation, ")");
            auto fn_ty = std::make_shared<ir::FunctionType>(ty, fparam_types);
            ty = fn_ty;
        }

        // Pointer suffix: i32* etc.
        while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
            next_token();
            ty = type_ctx_.pointer_to(ty);
        }
        return ty;
    }

    // Named type: %struct.X
    if (tok.type == TokenType::Punctuation && tok.text == "%") {
        next_token(); // consume '%'
        auto name_tok = next_token(); // struct name
        std::string full_name = "%" + name_tok.text;

        auto it = type_table_.find(name_tok.text);
        if (it != type_table_.end()) {
            auto ty = it->second;
            // Check for function-type suffix
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                next_token();
                std::vector<std::shared_ptr<ir::Type>> fparam_types;
                while (!(peek_token().type == TokenType::Punctuation && peek_token().text == ")")) {
                    skip_param_attrs();
                    fparam_types.push_back(parse_type());
                    if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") next_token();
                }
                expect(TokenType::Punctuation, ")");
                ty = std::make_shared<ir::FunctionType>(ty, fparam_types);
            }
            while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
                next_token();
                ty = type_ctx_.pointer_to(ty);
            }
            return ty;
        }

        // Try custom resolvers
        for (auto& resolver : type_resolvers_) {
            auto ty = resolver(full_name);
            if (ty) {
                while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
                    next_token();
                    ty = type_ctx_.pointer_to(ty);
                }
                return ty;
            }
        }

        // Unknown named type — create an opaque struct as placeholder (no warning)
        std::shared_ptr<ir::Type> ty = std::make_shared<ir::StructType>(
            std::vector<std::shared_ptr<ir::Type>>{}, false, name_tok.text);
        type_table_[name_tok.text] = ty;
        while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
            next_token();
            ty = type_ctx_.pointer_to(ty);
        }
        return ty;
    }

    // Array type: [N x type]
    if (tok.type == TokenType::Punctuation && tok.text == "[") {
        next_token(); // consume '['
        auto count_tok = next_token(); // count
        uint64_t count = std::stoull(count_tok.text);
        expect(TokenType::Identifier, "x"); // 'x' separator
        auto elem_type = parse_type();
        expect(TokenType::Punctuation, "]");
        std::shared_ptr<ir::Type> ty = std::make_shared<ir::ArrayType>(count, elem_type);
        while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
            next_token();
            ty = type_ctx_.pointer_to(ty);
        }
        return ty;
    }

    // Struct type: { type, type, ... }
    if (tok.type == TokenType::Punctuation && tok.text == "{") {
        next_token(); // consume '{'
        std::vector<std::shared_ptr<ir::Type>> fields;
        while (!(peek_token().type == TokenType::Punctuation && peek_token().text == "}")) {
            fields.push_back(parse_type());
            if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
                next_token();
            }
        }
        expect(TokenType::Punctuation, "}");
        std::shared_ptr<ir::Type> ty = std::make_shared<ir::StructType>(std::move(fields));
        while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
            next_token();
            ty = type_ctx_.pointer_to(ty);
        }
        return ty;
    }

    // Packed struct: <{ ... }>
    if (tok.type == TokenType::Punctuation && tok.text == "<") {
        // Distinguish between <{ ... }> (packed struct) and <N x type> (vector)
        if (pos_ + 1 < tokens_.size() && tokens_[pos_ + 1].type == TokenType::Punctuation &&
            tokens_[pos_ + 1].text == "{") {
            next_token(); // consume '<'
            expect(TokenType::Punctuation, "{");
            std::vector<std::shared_ptr<ir::Type>> fields;
            while (!(peek_token().type == TokenType::Punctuation && peek_token().text == "}")) {
                fields.push_back(parse_type());
                if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
                    next_token();
                }
            }
            expect(TokenType::Punctuation, "}");
            expect(TokenType::Punctuation, ">");
            std::shared_ptr<ir::Type> ty2 = std::make_shared<ir::StructType>(std::move(fields), true);
            while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
                next_token();
                ty2 = type_ctx_.pointer_to(ty2);
            }
            return ty2;
        }
        // Vector type: <N x type>
        if (pos_ + 1 < tokens_.size() && tokens_[pos_ + 1].type == TokenType::Number) {
            next_token(); // consume '<'
            auto count_tok = next_token();
            uint64_t count = std::stoull(count_tok.text);
            expect(TokenType::Identifier, "x");
            auto elem_type = parse_type();
            expect(TokenType::Punctuation, ">");
            std::shared_ptr<ir::Type> ty = type_ctx_.get_vector(elem_type, count);
            while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
                next_token();
                ty = type_ctx_.pointer_to(ty);
            }
            return ty;
        }
    }

    // Fallback: skip token and return void (no warning — might be valid IR we don't handle)
    next_token();
    return type_ctx_.void_type();
}

// ── Instruction parsing ────────────────────────────────────────────────────

std::shared_ptr<ir::Instruction> IRParser::parse_instruction(ir::Function& /*fn*/) {
    // Check if this is an assignment: %name = ...
    std::string result_name;
    if (peek_token().type == TokenType::Punctuation && peek_token().text == "%") {
        // Look ahead to see if there's an '='
        if (pos_ + 2 < tokens_.size() &&
            tokens_[pos_ + 2].type == TokenType::Punctuation &&
            tokens_[pos_ + 2].text == "=") {
            result_name = parse_name();
            match(TokenType::Punctuation, "=");
        }
    }

    auto tok = peek_token();

    // Skip metadata-only lines: !dbg !N, !metadata
    if (tok.type == TokenType::Punctuation && tok.text == "!") {
        skip_trailing_metadata();
        return nullptr;
    }

    // The opcode should be an identifier
    if (tok.type != TokenType::Identifier) {
        // Silently skip unknown tokens
        next_token();
        return nullptr;
    }

    std::string opcode_name = tok.text;

    // Binary operations
    if (binop_names.count(opcode_name)) {
        return parse_binop(opcode_name, result_name);
    }

    // Conversion operations
    if (conversion_names.count(opcode_name)) {
        return parse_conversion_op(opcode_name, result_name);
    }

    // Memory operations
    if (memop_names.count(opcode_name)) {
        return parse_memory_op(opcode_name, result_name);
    }

    // Terminators
    if (terminator_names.count(opcode_name)) {
        return parse_terminator(opcode_name);
    }

    // Comparison
    if (opcode_name == "icmp" || opcode_name == "fcmp") {
        next_token(); // consume 'icmp'/'fcmp'

        // Skip optional fast-math flags before the predicate (fcmp only)
        while (peek_token().type == TokenType::Keyword) {
            auto kw = peek_token().text;
            if (kw == "fast" || kw == "nnel" || kw == "nsz" || kw == "arcp" ||
                kw == "contract" || kw == "afn" || kw == "reassoc") {
                next_token();
            } else {
                break;
            }
        }

        // Parse predicate
        auto pred_tok = next_token();
        ir::CmpPredicate pred = ir::CmpPredicate::EQ; // default

        if (opcode_name == "icmp") {
            if (pred_tok.text == "eq")       pred = ir::CmpPredicate::EQ;
            else if (pred_tok.text == "ne")   pred = ir::CmpPredicate::NE;
            else if (pred_tok.text == "ugt")  pred = ir::CmpPredicate::UGT;
            else if (pred_tok.text == "uge")  pred = ir::CmpPredicate::UGE;
            else if (pred_tok.text == "ult")  pred = ir::CmpPredicate::ULT;
            else if (pred_tok.text == "ule")  pred = ir::CmpPredicate::ULE;
            else if (pred_tok.text == "sgt")  pred = ir::CmpPredicate::SGT;
            else if (pred_tok.text == "sge")  pred = ir::CmpPredicate::SGE;
            else if (pred_tok.text == "slt")  pred = ir::CmpPredicate::SLT;
            else if (pred_tok.text == "sle")  pred = ir::CmpPredicate::SLE;
        } else {
            // fcmp predicates
            if (pred_tok.text == "oeq")       pred = ir::CmpPredicate::FOEQ;
            else if (pred_tok.text == "ogt")  pred = ir::CmpPredicate::FOGT;
            else if (pred_tok.text == "oge")  pred = ir::CmpPredicate::FOGE;
            else if (pred_tok.text == "olt")  pred = ir::CmpPredicate::FOLT;
            else if (pred_tok.text == "ole")  pred = ir::CmpPredicate::FOLE;
            else if (pred_tok.text == "one")  pred = ir::CmpPredicate::FONE;
            else if (pred_tok.text == "ord")  pred = ir::CmpPredicate::FORD;
            else if (pred_tok.text == "uno")  pred = ir::CmpPredicate::FUNO;
            else if (pred_tok.text == "ueq")  pred = ir::CmpPredicate::FUEQ;
            else if (pred_tok.text == "ugt")  pred = ir::CmpPredicate::FUGT;
            else if (pred_tok.text == "uge")  pred = ir::CmpPredicate::FUGE;
            else if (pred_tok.text == "ult")  pred = ir::CmpPredicate::FULT;
            else if (pred_tok.text == "ule")  pred = ir::CmpPredicate::FULE;
            else if (pred_tok.text == "une")  pred = ir::CmpPredicate::FUNE;
            else if (pred_tok.text == "false") pred = ir::CmpPredicate::FFalse;
            else if (pred_tok.text == "true")  pred = ir::CmpPredicate::FTrue;
        }

        // Parse the type and operands
        auto ty = parse_type();
        auto lhs = parse_value(ty);
        expect(TokenType::Punctuation, ",");
        auto rhs = parse_value(ty);

        auto inst = ir::inst::make_icmp(pred, lhs, rhs, result_name);
        skip_trailing_metadata();
        return inst;
    }

    // Phi node: %x = phi i32 [ %val1, %bb1 ], [ %val2, %bb2 ]
    if (opcode_name == "phi") {
        next_token(); // consume 'phi'
        auto ty = parse_type();

        auto inst = ir::inst::make_phi(ty, result_name);

        // Parse incoming values: [ val, %bb ], ...
        // Record the incoming block labels in the "phi_blocks" metadata
        // (comma-separated, in operand order) — that is where
        // Instruction::to_string() reads them back from for IR round-trip.
        std::string phi_blocks;
        while (peek_token().type == TokenType::Punctuation && peek_token().text == "[") {
            next_token(); // consume '['
            auto val = parse_value(ty);
            expect(TokenType::Punctuation, ",");
            auto bb_name = parse_name(); // block name
            expect(TokenType::Punctuation, "]");
            inst->add_operand(val);
            if (!phi_blocks.empty()) phi_blocks += ',';
            phi_blocks += bb_name;
            if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
                next_token();
            }
        }
        if (!phi_blocks.empty()) {
            inst->set_metadata("phi_blocks", phi_blocks);
        }

        skip_trailing_metadata();
        return inst;
    }

    // Call: %x = call [tail] [cconv] [ret attrs] ret_type [@func](args)
    if (opcode_name == "call") {
        next_token(); // consume 'call'

        // Skip tail/fast/musttail/notail and calling convention keywords
        while (peek_token().type == TokenType::Keyword) {
            auto kw = peek_token().text;
            if (kw == "tail" || kw == "fast" || kw == "musttail" || kw == "notail" ||
                kw == "nnel" ||
                kw == "ccc" || kw == "fastcc" || kw == "coldcc" ||
                kw == "webkit_jscc" || kw == "anyregcc" ||
                kw == "preserve_mostcc" || kw == "preserve_allcc" ||
                kw == "swiftcc" || kw == "swifttailcc" ||
                kw == "cfguard_checkcc" || kw == "c") {
                next_token();
            } else {
                // Could be a return attribute (signext, zeroext, inreg, etc.)
                next_token();
                if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                    skip_balanced('(', ')');
                }
            }
        }

        auto ret_type = parse_type();

        // Check for explicit function type: ret_type (param_types...)*
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
            next_token(); // consume '('
            while (!(peek_token().type == TokenType::Punctuation && peek_token().text == ")")) {
                skip_param_attrs();
                if (peek_token().type == TokenType::Punctuation && peek_token().text == "...") {
                    next_token();
                    if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") next_token();
                    continue;
                }
                parse_type();
                if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") next_token();
            }
            expect(TokenType::Punctuation, ")");
            // Consume pointer indirection
            while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
                next_token();
            }
        }

        std::string callee = parse_name();

        // Parse arguments
        expect(TokenType::Punctuation, "(");
        std::vector<std::shared_ptr<ir::Value>> call_args;
        while (!(peek_token().type == TokenType::Punctuation && peek_token().text == ")")) {
            // Skip parameter attributes
            skip_param_attrs();
            auto arg_type = parse_type();
            // Attributes can also appear between type and value
            skip_param_attrs();
            auto arg_val = parse_value(arg_type);
            call_args.push_back(arg_val);
            if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
                next_token();
            }
        }
        expect(TokenType::Punctuation, ")");

        auto inst = ir::inst::make_call(ret_type, callee, std::move(call_args), result_name);
        skip_trailing_metadata();
        return inst;
    }

    // Select: %x = select i1 %cond, i32 %true, i32 %false
    if (opcode_name == "select") {
        next_token(); // consume 'select'

        // Skip optional fast-math flags
        while (peek_token().type == TokenType::Keyword) {
            next_token();
        }

        auto cond_type = parse_type();
        auto cond = parse_value(cond_type);
        expect(TokenType::Punctuation, ",");
        auto true_type = parse_type();
        auto true_val = parse_value(true_type);
        expect(TokenType::Punctuation, ",");
        auto false_type = parse_type();
        auto false_val = parse_value(false_type);

        auto inst = ir::inst::make_select(cond, true_val, false_val, result_name);
        skip_trailing_metadata();
        return inst;
    }

    // Switch: switch i32 %val, label %default [i32 1, label %bb1 i32 2, label %bb2]
    if (opcode_name == "switch") {
        next_token(); // consume 'switch'
        auto val_type = parse_type();
        auto val = parse_value(val_type);
        expect(TokenType::Punctuation, ",");
        // default label
        match(TokenType::Keyword, "label"); // optional 'label' keyword
        auto default_bb = parse_name();

        auto inst = std::make_shared<ir::Instruction>(ir::Opcode::SwitchInst,
                                                       type_ctx_.void_type(), result_name);
        inst->add_operand(val);

        // Parse cases: [ i32 N, label %bb ]
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "[") {
            next_token(); // consume '['
            while (!(peek_token().type == TokenType::Punctuation && peek_token().text == "]")) {
                auto case_type = parse_type();
                auto case_val = parse_value(case_type);
                expect(TokenType::Punctuation, ",");
                match(TokenType::Keyword, "label");
                auto case_bb = parse_name();
                inst->add_operand(case_val);
                if (peek_token().type == TokenType::Punctuation && peek_token().text == "]") break;
            }
            expect(TokenType::Punctuation, "]");
        }
        skip_trailing_metadata();
        return inst;
    }

    // ExtractElement: %x = extractelement <N x ty> %vec, iM %idx
    if (opcode_name == "extractelement") {
        next_token(); // consume 'extractelement'
        auto vec_ty = parse_type();
        auto vec = parse_value(vec_ty);
        expect(TokenType::Punctuation, ",");
        auto idx_ty = parse_type();
        auto idx = parse_value(idx_ty);
        auto inst = ir::inst::make_extractelement(vec, idx, result_name);
        skip_trailing_metadata();
        return inst;
    }

    // InsertElement: %x = insertelement <N x ty> %vec, ty %elem, iM %idx
    if (opcode_name == "insertelement") {
        next_token(); // consume 'insertelement'
        auto vec_ty = parse_type();
        auto vec = parse_value(vec_ty);
        expect(TokenType::Punctuation, ",");
        auto elem_ty = parse_type();
        auto elem = parse_value(elem_ty);
        expect(TokenType::Punctuation, ",");
        auto idx_ty = parse_type();
        auto idx = parse_value(idx_ty);
        auto inst = ir::inst::make_insertelement(vec, elem, idx, result_name);
        skip_trailing_metadata();
        return inst;
    }

    // ShuffleVector: %x = shufflevector <N x ty> %a, <N x ty> %b, <K x i32> <mask>
    if (opcode_name == "shufflevector") {
        next_token(); // consume 'shufflevector'
        auto lhs_ty = parse_type();
        auto lhs = parse_value(lhs_ty);
        expect(TokenType::Punctuation, ",");
        auto rhs_ty = parse_type();
        auto rhs = parse_value(rhs_ty);
        expect(TokenType::Punctuation, ",");
        auto mask_ty = parse_type();
        auto mask = parse_value(mask_ty);
        auto inst = ir::inst::make_shufflevector(lhs, rhs, mask, result_name);
        skip_trailing_metadata();
        return inst;
    }

    // Invoke: invoke ... to label %bb unwind label %bb
    if (opcode_name == "invoke") {
        next_token(); // consume 'invoke'

        // Skip calling convention / return attrs
        while (peek_token().type == TokenType::Keyword) {
            next_token();
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                skip_balanced('(', ')');
            }
        }

        auto ret_type = parse_type();

        // Check for explicit function type
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
            next_token();
            while (!(peek_token().type == TokenType::Punctuation && peek_token().text == ")")) {
                skip_param_attrs();
                if (peek_token().type == TokenType::Punctuation && peek_token().text == "...") {
                    next_token();
                    if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") next_token();
                    continue;
                }
                parse_type();
                if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") next_token();
            }
            expect(TokenType::Punctuation, ")");
            while (peek_token().type == TokenType::Punctuation && peek_token().text == "*") {
                next_token();
            }
        }

        std::string callee = parse_name();

        expect(TokenType::Punctuation, "(");
        std::vector<std::shared_ptr<ir::Value>> invoke_args;
        while (!(peek_token().type == TokenType::Punctuation && peek_token().text == ")")) {
            skip_param_attrs();
            auto arg_type = parse_type();
            // Attributes can also appear between type and value
            skip_param_attrs();
            auto arg_val = parse_value(arg_type);
            invoke_args.push_back(arg_val);
            if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") next_token();
        }
        expect(TokenType::Punctuation, ")");

        auto inst = ir::inst::make_call(ret_type, callee, std::move(invoke_args), result_name);

        // "to label %bb unwind label %bb"
        match(TokenType::Keyword, "to");
        match(TokenType::Keyword, "label");
        parse_name(); // normal dest
        match(TokenType::Keyword, "unwind");
        match(TokenType::Keyword, "label");
        parse_name(); // unwind dest

        skip_trailing_metadata();
        return inst;
    }

    // Unknown instruction — skip silently
    next_token(); // consume the opcode

    // Consume remaining tokens until next instruction boundary
    while (peek_token().type != TokenType::Eof) {
        auto next = peek_token();
        // Stop at closing brace
        if (next.type == TokenType::Punctuation && next.text == "}") break;
        // Stop at a new label
        if (next.type == TokenType::Identifier &&
            pos_ + 1 < tokens_.size() &&
            tokens_[pos_ + 1].type == TokenType::Punctuation &&
            tokens_[pos_ + 1].text == ":") break;
        // Stop at a new known instruction
        if (next.type == TokenType::Identifier && is_known_instruction(next.text)) break;
        // Stop at %result = pattern
        if (next.type == TokenType::Punctuation && next.text == "%" &&
            pos_ + 2 < tokens_.size() &&
            tokens_[pos_ + 2].type == TokenType::Punctuation &&
            tokens_[pos_ + 2].text == "=") {
            break;
        }
        next_token();
    }

    return nullptr;
}

std::shared_ptr<ir::Instruction> IRParser::parse_binop(const std::string& opcode_name,
                                                         const std::string& result_name) {
    next_token(); // consume opcode

    // Map opcode name to Opcode enum
    ir::Opcode opc;
    if (opcode_name == "add")           opc = ir::Opcode::Add;
    else if (opcode_name == "sub")      opc = ir::Opcode::Sub;
    else if (opcode_name == "mul")      opc = ir::Opcode::Mul;
    else if (opcode_name == "udiv")     opc = ir::Opcode::UDiv;
    else if (opcode_name == "sdiv")     opc = ir::Opcode::SDiv;
    else if (opcode_name == "urem")     opc = ir::Opcode::URem;
    else if (opcode_name == "srem")     opc = ir::Opcode::SRem;
    else if (opcode_name == "fadd")     opc = ir::Opcode::FAdd;
    else if (opcode_name == "fsub")     opc = ir::Opcode::FSub;
    else if (opcode_name == "fmul")     opc = ir::Opcode::FMul;
    else if (opcode_name == "fdiv")     opc = ir::Opcode::FDiv;
    else if (opcode_name == "frem")     opc = ir::Opcode::FRem;
    else if (opcode_name == "and")      opc = ir::Opcode::And;
    else if (opcode_name == "or")       opc = ir::Opcode::Or;
    else if (opcode_name == "xor")      opc = ir::Opcode::Xor;
    else if (opcode_name == "shl")      opc = ir::Opcode::Shl;
    else if (opcode_name == "lshr")     opc = ir::Opcode::LShr;
    else if (opcode_name == "ashr")     opc = ir::Opcode::AShr;
    else {
        return nullptr;
    }

    // Parse optional flags: nuw, nsw, exact, fast-math flags, nneg, disjoint
    ir::BinOpFlags flags;
    while (peek_token().type == TokenType::Keyword) {
        auto flag = peek_token().text;
        if (flag == "nuw")       { flags.nuw = true; next_token(); }
        else if (flag == "nsw")  { flags.nsw = true; next_token(); }
        else if (flag == "exact"){ flags.exact = true; next_token(); }
        else if (flag == "nneg")      { next_token(); }
        else if (flag == "disjoint")  { next_token(); }
        else if (flag == "fast")      { next_token(); }
        else if (flag == "nnel")      { next_token(); }
        else if (flag == "contract")  { next_token(); }
        else if (flag == "afn")       { next_token(); }
        else if (flag == "reassoc")   { next_token(); }
        else if (flag == "nsz")       { next_token(); }
        else if (flag == "arcp")      { next_token(); }
        else break;
    }

    // Parse type
    auto ty = parse_type();
    auto lhs = parse_value(ty);
    expect(TokenType::Punctuation, ",");
    auto rhs = parse_value(ty);

    auto inst = std::make_shared<ir::Instruction>(opc, ty, result_name);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    inst->binop_flags() = flags;

    skip_trailing_metadata();
    return inst;
}

std::shared_ptr<ir::Instruction> IRParser::parse_memory_op(const std::string& opcode_name,
                                                              const std::string& result_name) {
    if (opcode_name == "alloca") {
        next_token(); // consume 'alloca'

        // Parse allocated type
        auto allocated_type = parse_type();

        // Optional: , i32 N (count)
        std::shared_ptr<ir::Value> count;
        if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
            next_token();
            // Check for 'align' keyword (not a count)
            if (peek_token().type == TokenType::Keyword && peek_token().text == "align") {
                next_token();
                auto align_tok = next_token();
                unsigned align = static_cast<unsigned>(std::stoul(align_tok.text));
                auto ptr_type = type_ctx_.pointer_to(allocated_type);
                return ir::inst::make_alloca(allocated_type, result_name, align);
            }
            auto count_type = parse_type();
            count = parse_value(count_type);
        }

        unsigned align = 0;
        // Optional: , align N
        if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
            next_token();
            if (peek_token().type == TokenType::Keyword && peek_token().text == "align") {
                next_token();
                auto align_tok = next_token();
                align = static_cast<unsigned>(std::stoul(align_tok.text));
            }
        }

        auto ptr_type = type_ctx_.pointer_to(allocated_type);
        auto inst = ir::inst::make_alloca(allocated_type, result_name, align);
        if (count) inst->add_operand(count);

        skip_trailing_metadata();
        return inst;
    }

    if (opcode_name == "load") {
        next_token(); // consume 'load'

        // Optional: atomic
        if (peek_token().type == TokenType::Keyword && peek_token().text == "atomic") {
            next_token();
        }

        // Parse type: the type being loaded (not the pointer type)
        auto loaded_type = parse_type();

        // Comma then pointer type + value
        expect(TokenType::Punctuation, ",");
        auto ptr_type = parse_type();
        auto ptr = parse_value(ptr_type);

        unsigned align = 0;
        bool is_volatile = false;

        // Parse optional attributes: , align N, , volatile, , acquire, etc.
        while (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
            next_token();
            if (peek_token().type == TokenType::Keyword && peek_token().text == "align") {
                next_token();
                auto align_tok = next_token();
                align = static_cast<unsigned>(std::stoul(align_tok.text));
            } else if (peek_token().type == TokenType::Keyword && peek_token().text == "volatile") {
                is_volatile = true;
                next_token();
            } else if (peek_token().type == TokenType::Keyword) {
                // Skip atomic ordering, syncscope, etc.
                next_token();
                if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                    skip_balanced('(', ')');
                }
            } else {
                break;
            }
        }

        auto inst = ir::inst::make_load(loaded_type, ptr, result_name, align);
        if (is_volatile) inst->set_volatile(true);

        skip_trailing_metadata();
        return inst;
    }

    if (opcode_name == "store") {
        next_token(); // consume 'store'

        // Optional: atomic
        if (peek_token().type == TokenType::Keyword && peek_token().text == "atomic") {
            next_token();
        }

        // Parse value type
        auto val_type = parse_type();
        auto val = parse_value(val_type);
        expect(TokenType::Punctuation, ",");
        auto ptr_type = parse_type();
        auto ptr = parse_value(ptr_type);

        unsigned align = 0;
        bool is_volatile = false;

        // Parse optional attributes
        while (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
            next_token();
            if (peek_token().type == TokenType::Keyword && peek_token().text == "align") {
                next_token();
                auto align_tok = next_token();
                align = static_cast<unsigned>(std::stoul(align_tok.text));
            } else if (peek_token().type == TokenType::Keyword && peek_token().text == "volatile") {
                is_volatile = true;
                next_token();
            } else if (peek_token().type == TokenType::Keyword) {
                // Skip atomic ordering, syncscope, etc.
                next_token();
                if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                    skip_balanced('(', ')');
                }
            } else {
                break;
            }
        }

        auto inst = ir::inst::make_store(val, ptr, align);
        if (is_volatile) inst->set_volatile(true);

        skip_trailing_metadata();
        return inst;
    }

    if (opcode_name == "getelementptr") {
        next_token(); // consume 'getelementptr'

        // Optional: inbounds
        bool inbounds = false;
        if (peek_token().type == TokenType::Keyword && peek_token().text == "inbounds") {
            inbounds = true;
            next_token();
        }

        // Parse the element type
        auto elem_type = parse_type();

        // Comma then base pointer
        expect(TokenType::Punctuation, ",");
        auto ptr_type = parse_type();
        auto ptr = parse_value(ptr_type);

        // Parse indices. The base pointer is NOT an index — make_gep
        // adds it as operand 0 itself.
        std::vector<std::shared_ptr<ir::Value>> indices;

        while (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
            next_token();
            auto idx_type = parse_type();
            auto idx = parse_value(idx_type);
            indices.push_back(idx);
        }

        // Result type is a pointer to elem_type
        auto result_type = type_ctx_.pointer_to(elem_type);
        auto inst = ir::inst::make_gep(result_type, ptr, std::move(indices), result_name);

        skip_trailing_metadata();
        return inst;
    }

    // fence, cmpxchg, atomicrmw — skip silently, return nullptr
    next_token(); // consume opcode

    // Skip remaining tokens until instruction boundary
    while (peek_token().type != TokenType::Eof) {
        auto next = peek_token();
        if (next.type == TokenType::Punctuation && next.text == "}") break;
        if (next.type == TokenType::Identifier &&
            pos_ + 1 < tokens_.size() &&
            tokens_[pos_ + 1].type == TokenType::Punctuation &&
            tokens_[pos_ + 1].text == ":") break;
        if (next.type == TokenType::Identifier && is_known_instruction(next.text)) break;
        if (next.type == TokenType::Punctuation && next.text == "%" &&
            pos_ + 2 < tokens_.size() &&
            tokens_[pos_ + 2].type == TokenType::Punctuation &&
            tokens_[pos_ + 2].text == "=") break;
        next_token();
    }

    return nullptr;
}

std::shared_ptr<ir::Instruction> IRParser::parse_terminator(const std::string& opcode_name) {
    if (opcode_name == "ret") {
        next_token(); // consume 'ret'

        // ret void | ret type %val
        auto ty = parse_type();
        if (ty->is_void()) {
            skip_trailing_metadata();
            return ir::inst::make_ret_void();
        }

        auto val = parse_value(ty);
        skip_trailing_metadata();
        return ir::inst::make_ret(val);
    }

    if (opcode_name == "br") {
        next_token(); // consume 'br'

        // Check if next token is 'label' (unconditional) or a type (conditional)
        if (peek_token().type == TokenType::Keyword && peek_token().text == "label") {
            // Unconditional branch
            next_token(); // consume 'label'
            auto dest = parse_name();
            auto inst = ir::inst::make_br_uncond(dest);
            skip_trailing_metadata();
            return inst;
        }

        // Conditional branch
        auto cond_type = parse_type();
        auto cond = parse_value(cond_type);
        expect(TokenType::Punctuation, ",");
        match(TokenType::Keyword, "label"); // optional 'label' keyword
        auto true_bb = parse_name();
        expect(TokenType::Punctuation, ",");
        match(TokenType::Keyword, "label");
        auto false_bb = parse_name();
        auto inst = ir::inst::make_br(cond, true_bb, false_bb);
        skip_trailing_metadata();
        return inst;
    }

    if (opcode_name == "unreachable") {
        next_token(); // consume 'unreachable'
        auto inst = std::make_shared<ir::Instruction>(ir::Opcode::Unreachable,
                                                       type_ctx_.void_type(), "");
        skip_trailing_metadata();
        return inst;
    }

    if (opcode_name == "resume") {
        next_token(); // consume 'resume'
        auto ty = parse_type();
        auto val = parse_value(ty);
        auto inst = std::make_shared<ir::Instruction>(ir::Opcode::Resume,
                                                       ty, "");
        inst->add_operand(val);
        skip_trailing_metadata();
        return inst;
    }

    if (opcode_name == "switch") {
        // Handled in parse_instruction
        next_token(); // consume 'switch'
        return nullptr;
    }

    if (opcode_name == "invoke") {
        // Handled in parse_instruction
        next_token(); // consume 'invoke'
        return nullptr;
    }

    // Unknown terminator — skip silently
    next_token();
    return nullptr;
}

std::shared_ptr<ir::Instruction> IRParser::parse_conversion_op(const std::string& opcode_name,
                                                                  const std::string& result_name) {
    next_token(); // consume opcode

    ir::Opcode opc;
    if (opcode_name == "trunc")              opc = ir::Opcode::Trunc;
    else if (opcode_name == "zext")          opc = ir::Opcode::ZExt;
    else if (opcode_name == "sext")          opc = ir::Opcode::SExt;
    else if (opcode_name == "fptrunc")       opc = ir::Opcode::FPTrunc;
    else if (opcode_name == "fpext")         opc = ir::Opcode::FPExt;
    else if (opcode_name == "fptoui")        opc = ir::Opcode::FPToUI;
    else if (opcode_name == "fptosi")        opc = ir::Opcode::FPToSI;
    else if (opcode_name == "uitofp")        opc = ir::Opcode::UIToFP;
    else if (opcode_name == "sitofp")        opc = ir::Opcode::SIToFP;
    else if (opcode_name == "ptrtoint")      opc = ir::Opcode::PtrToInt;
    else if (opcode_name == "inttoptr")      opc = ir::Opcode::IntToPtr;
    else if (opcode_name == "bitcast")       opc = ir::Opcode::BitCast;
    else if (opcode_name == "addrspacecast") opc = ir::Opcode::AddrSpaceCast;
    else {
        return nullptr;
    }

    // Parse: source_type %val to dest_type
    auto src_type = parse_type();
    auto val = parse_value(src_type);
    match(TokenType::Keyword, "to");
    auto dest_type = parse_type();

    auto inst = std::make_shared<ir::Instruction>(opc, dest_type, result_name);
    inst->add_operand(val);

    skip_trailing_metadata();
    return inst;
}

// ── Value/operand parsing ──────────────────────────────────────────────────

std::shared_ptr<ir::Value> IRParser::parse_value(std::shared_ptr<ir::Type> expected_type) {
    auto tok = peek_token();

    // Named value: %name
    if (tok.type == TokenType::Punctuation && tok.text == "%") {
        auto name = parse_name();
        auto it = value_table_.find(name);
        if (it != value_table_.end()) {
            return it->second;
        }
        // Create a placeholder and register it
        auto val = std::make_shared<ir::Value>(expected_type, name);
        value_table_[name] = val;
        return val;
    }

    // Global value: @name
    if (tok.type == TokenType::Punctuation && tok.text == "@") {
        auto name = parse_name();
        auto ptr_type = expected_type->is_pointer() ? expected_type :
                        type_ctx_.pointer_to(expected_type);
        // Reuse the same Value object for repeated references to the
        // same global within this function (mirrors the %name dedup
        // just above) — keyed with a "@" prefix in value_table_ so it
        // can never collide with a local SSA name of the same text.
        const std::string key = "@" + name;
        auto it = value_table_.find(key);
        if (it != value_table_.end()) {
            return it->second;
        }
        auto val = std::make_shared<ir::Value>(ptr_type, name, /*is_global=*/true);
        value_table_[key] = val;
        return val;
    }

    // Numeric constant. The expected type decides the interpretation:
    // an FP-typed context yields a ConstantFP, an integer context a
    // ConstantInt.
    if (tok.type == TokenType::Number) {
        next_token();
        if (expected_type->is_float() || expected_type->is_double()) {
            double dval = 0.0;
            const std::string& t = tok.text;
            try {
                if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
                    // LLVM hex FP literals are the RAW IEEE-754 bits of a
                    // double (0x3FF0000000000000 == 1.0), not a hex float.
                    uint64_t fp_bits = std::stoull(t.substr(2), nullptr, 16);
                    static_assert(sizeof(fp_bits) == sizeof(dval),
                                  "bit reinterpretation requires 64-bit double");
                    std::memcpy(&dval, &fp_bits, sizeof(dval));
                } else {
                    dval = std::stod(t);
                }
            } catch (...) {
                dval = 0.0;
            }
            return std::make_shared<ir::ConstantFP>(expected_type, dval);
        }
        try {
            int64_t ival = std::stoll(tok.text, nullptr, 0);
            unsigned bits = expected_type->is_integer() ? expected_type->bit_width() : 32;
            return ir::ConstantInt::get(type_ctx_, ival, bits);
        } catch (...) {
            return ir::ConstantInt::get(type_ctx_, 0, 32);
        }
    }

    // Vector constant literal: <i32 1, i32 2, ...>
    if (tok.type == TokenType::Punctuation && tok.text == "<" &&
        expected_type->is_vector()) {
        next_token(); // consume '<'
        std::vector<std::shared_ptr<ir::Value>> elems;
        while (!(peek_token().type == TokenType::Punctuation && peek_token().text == ">")) {
            auto elem_ty = parse_type();
            elems.push_back(parse_value(elem_ty));
            if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
                next_token();
            }
        }
        expect(TokenType::Punctuation, ">");
        auto vec_ty = std::static_pointer_cast<ir::VectorType>(expected_type);
        return std::make_shared<ir::ConstantVector>(vec_ty, std::move(elems));
    }

    // Value keywords (kept as Identifiers for parse_value to handle)
    if (tok.type == TokenType::Identifier) {
        if (tok.text == "true") {
            next_token();
            return ir::ConstantInt::get(type_ctx_, 1, 1);
        }
        if (tok.text == "false") {
            next_token();
            return ir::ConstantInt::get(type_ctx_, 0, 1);
        }
        if (tok.text == "null") {
            next_token();
            auto ptr_type = std::dynamic_pointer_cast<ir::PointerType>(expected_type);
            if (!ptr_type) ptr_type = type_ctx_.pointer_to(type_ctx_.void_type());
            return std::make_shared<ir::ConstantPointerNull>(ptr_type);
        }
        if (tok.text == "undef") {
            next_token();
            return std::make_shared<ir::UndefValue>(expected_type);
        }
        if (tok.text == "poison") {
            next_token();
            return std::make_shared<ir::PoisonValue>(expected_type);
        }
        if (tok.text == "zeroinitializer") {
            next_token();
            if (expected_type->is_integer()) {
                unsigned bits = expected_type->bit_width();
                return ir::ConstantInt::get(type_ctx_, 0, static_cast<unsigned>(bits));
            }
            if (expected_type->is_vector()) {
                auto vec_ty = std::static_pointer_cast<ir::VectorType>(expected_type);
                if (vec_ty->element_type()->is_integer()) {
                    std::vector<int64_t> lanes(vec_ty->count(), 0);
                    return ir::ConstantVector::get_int_lanes(
                        type_ctx_, lanes,
                        static_cast<unsigned>(vec_ty->element_type()->bit_width()));
                }
            }
            if (expected_type->is_float()) {
                return std::make_shared<ir::ConstantFP>(expected_type, 0.0);
            }
            if (expected_type->is_double()) {
                return std::make_shared<ir::ConstantFP>(expected_type, 0.0);
            }
            return std::make_shared<ir::UndefValue>(expected_type);
        }
        if (tok.text == "none") {
            next_token();
            return std::make_shared<ir::UndefValue>(expected_type);
        }
    }

    // Could be a complex constant expression, aggregate constant, etc.
    // Silently skip and return an undef (no warning)
    next_token();
    return std::make_shared<ir::UndefValue>(expected_type);
}

// ── Helpers ────────────────────────────────────────────────────────────────

std::shared_ptr<ir::Type> IRParser::resolve_type(const std::string& name) {
    // Check built-in types
    if (name == "void") return type_ctx_.void_type();
    if (name == "float") return type_ctx_.float_type();
    if (name == "double") return type_ctx_.double_type();
    if (name.size() > 1 && name[0] == 'i' &&
        std::all_of(name.begin() + 1, name.end(), ::isdigit)) {
        return type_ctx_.get_int(static_cast<unsigned>(std::stoul(name.substr(1))));
    }

    // Check type table
    auto it = type_table_.find(name);
    if (it != type_table_.end()) return it->second;

    // Try custom resolvers
    for (auto& resolver : type_resolvers_) {
        auto ty = resolver(name);
        if (ty) return ty;
    }

    return nullptr;
}

std::string IRParser::parse_name() {
    // Parse %name or @name
    auto prefix = next_token(); // '%' or '@'
    if (prefix.type != TokenType::Punctuation ||
        (prefix.text != "%" && prefix.text != "@")) {
        throw ParseError("Expected '%' or '@' for name, got '" + prefix.text +
                         "' at line " + std::to_string(prefix.loc.line));
    }

    auto name_tok = next_token();
    std::string result = name_tok.text;

    // Handle numeric names (like %0, %1 for unnamed values)
    if (name_tok.type == TokenType::Number) {
        // Numeric names are valid in LLVM IR
    }

    // Handle quoted names (like @"foo bar")
    if (name_tok.type == TokenType::String) {
        // Strip the quotes from the name
        if (result.size() >= 2 && result.front() == '"' && result.back() == '"') {
            result = result.substr(1, result.size() - 2);
        }
    }

    // Handle keyword names (like %default, %private — keywords can be used as SSA names)
    if (name_tok.type == TokenType::Keyword) {
        // The keyword text is the name
    }

    // Handle Type tokens used as names (like %i32 — rare but valid in some contexts)
    if (name_tok.type == TokenType::Type) {
        // The type text is the name
    }

    return result; // Store without prefix in value_table_
}

// ── parse_llvm_module_flags() ──────────────────────────────────────────────
// Consume: !llvm.module.flags = !{ !N, !M, ... }
// The individual entries (!N = !{ i32 <behavior>, !"<key>", <value> }) are
// parsed separately by try_parse_module_flag_entry().
void IRParser::parse_llvm_module_flags(std::shared_ptr<ir::Module> mod) {
    // Skip: !llvm.module.flags = !{ ... }
    next_token(); // consume '!'
    next_token(); // consume 'llvm.module.flags'
    if (peek_token().type == TokenType::Punctuation && peek_token().text == "=") {
        next_token(); // consume '='
    }
    if (peek_token().type == TokenType::Punctuation && peek_token().text == "!") {
        next_token(); // consume '!'
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "{") {
            skip_balanced('{', '}');
        }
    }
    // The collection line is informational only — the actual flag data was
    // already captured by try_parse_module_flag_entry().
}

// ── try_parse_module_flag_entry() ──────────────────────────────────────────
// Attempt to parse: !N = !{ i32 <behavior>, !"<key>", <value> }
//
// Returns:
//   - true  if a valid module flag entry was found AND captured.
//   - true  if the construct parsed as `!N = !{...}` (or `!N = !{ !... }`)
//           but did NOT match the module-flag shape — in that case we
//           still consumed the tokens, but we ALSO stored the metadata as
//           a generic numbered metadata def so it round-trips.  The
//           dispatcher's follow-up parse_metadata_def() is therefore not
//           needed (and would do nothing, since pos_ has advanced past
//           the construct).
//   - false if the input did not even look like `!N = !{...}`; in that
//           case pos_ is rewound to saved_pos so the dispatcher can try
//           other parsers.
bool IRParser::try_parse_module_flag_entry(std::shared_ptr<ir::Module> mod) {
    // Current token is '!', next should be a Number
    if (!(peek_token().type == TokenType::Punctuation && peek_token().text == "!"))
        return false;
    size_t saved_pos = pos_;
    next_token(); // consume '!'

    if (peek_token().type != TokenType::Number) {
        pos_ = saved_pos;
        return false;
    }
    auto id_tok = peek_token();
    std::string id = id_tok.text;
    next_token(); // consume the number (metadata ID)

    if (!(peek_token().type == TokenType::Punctuation && peek_token().text == "=")) {
        pos_ = saved_pos;
        return false;
    }
    next_token(); // consume '='

    // Optional `distinct` keyword before the body.
    std::string distinct_prefix;
    if (peek_token().type == TokenType::Identifier && peek_token().text == "distinct") {
        distinct_prefix = "distinct ";
        next_token();
    }

    // Expect !{ ... }
    if (!(peek_token().type == TokenType::Punctuation && peek_token().text == "!")) {
        pos_ = saved_pos;
        return false;
    }
    next_token(); // consume '!'

    if (!(peek_token().type == TokenType::Punctuation && peek_token().text == "{")) {
        pos_ = saved_pos;
        return false;
    }
    next_token(); // consume '{'

    // Parse: i32 <behavior>, !"<key>", <value>
    ir::ModuleFlag flag;
    flag.behavior = 0;
    bool found = false;

    // First element: i32 <behavior>
    // Note: the lexer classifies `i32`/`i64` as TokenType::Type (not
    // Identifier), so we must accept either type here.
    //
    // IMPORTANT: a module flag's first element MUST be `i32 <number>` (or
    // `i64 <number>`).  If we don't see that, this `!N = !{...}` is NOT a
    // module flag — it's a plain metadata def (e.g. `!2 = !{!"clang..."}`).
    // Rewind so parse_metadata_def() can capture it cleanly.
    if (!((peek_token().type == TokenType::Identifier ||
           peek_token().type == TokenType::Type) &&
          (peek_token().text == "i32" || peek_token().text == "i64"))) {
        pos_ = saved_pos;
        return false;
    }
    next_token(); // consume the type keyword
    if (peek_token().type == TokenType::Number) {
        flag.behavior = static_cast<unsigned>(std::stoul(peek_token().text));
        next_token(); // consume behavior number
    } else {
        // `i32` not followed by a number — also not a valid module flag.
        pos_ = saved_pos;
        return false;
    }

    // Comma
    if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
        next_token(); // consume ','
    }

    // Second element: !"<key>"
    if (peek_token().type == TokenType::Punctuation && peek_token().text == "!") {
        next_token(); // consume '!'
        if (peek_token().type == TokenType::String) {
            std::string key = peek_token().text;
            // Strip surrounding quotes
            if (key.size() >= 2 && key.front() == '"' && key.back() == '"') {
                key = key.substr(1, key.size() - 2);
            }
            flag.key = key;
            next_token(); // consume the string
            found = true;
        }
    }

    // Comma
    if (peek_token().type == TokenType::Punctuation && peek_token().text == ",") {
        next_token(); // consume ','
    }

    // Third element: <value> — consume everything until closing '}'
    std::string value;
    int brace_depth = 1; // already inside the '{'
    while (peek_token().type != TokenType::Eof && brace_depth > 0) {
        auto tok = peek_token();
        if (tok.type == TokenType::Punctuation && tok.text == "{") brace_depth++;
        if (tok.type == TokenType::Punctuation && tok.text == "}") brace_depth--;
        if (brace_depth <= 0) break;
        // Add a space separator before the token unless:
        //   - value is empty
        //   - last char is already a space or '{'
        //   - the token is ',' (no space before commas)
        if (!value.empty() && value.back() != ' ' && value.back() != '{' &&
            tok.text != "," && tok.text != "}") {
            value += ' ';
        }
        if (tok.type == TokenType::Punctuation && tok.text == "!") {
            // `!N` or `!"string"` — append '!' and the following token
            // WITHOUT a space, matching how LLVM IR is normally written.
            value += '!';
            next_token();
            auto after = peek_token();
            // Don't insert a space between '!' and its operand.
            value += after.text;
            next_token();
            continue;
        }
        value += tok.text;
        next_token();
    }

    // Trim leading/trailing whitespace from value
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
        value.erase(value.begin());
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t'))
        value.pop_back();

    flag.value = value;

    // Consume the closing '}'
    if (peek_token().type == TokenType::Punctuation && peek_token().text == "}") {
        next_token(); // consume '}'
    }

    if (found && flag.behavior >= 1 && flag.behavior <= 5) {
        mod->add_module_flag(flag);
        return true;
    }

    // We consumed `!N = [distinct] !{...}` but it wasn't a valid module
    // flag (e.g. `!2 = !{!"clang version 18.0.0"}` — no i32 behavior
    // prefix).  Don't drop it on the floor — store it as a generic
    // numbered metadata def so it round-trips.  Reconstruct the body
    // as `[distinct ]!{<value>}` with no leading/trailing space inside
    // the braces when value is non-empty (matches clang's emission of
    // `!{!"clang version 18.0.0"}` rather than `!{ !"clang version 18.0.0" }`).
    std::string body = distinct_prefix;
    body += '!';
    body += '{';
    if (!value.empty()) {
        body += value;
    }
    body += '}';
    mod->add_metadata_def(id, body);

    // We DID consume the construct — return true so the dispatcher's
    // follow-up parse_metadata_def() doesn't try to re-parse what's
    // already gone.  (Returning true here is the "consumed but not a
    // module flag" case described in the function header comment.)
    return true;
}

// ── collect_function_attrs() ────────────────────────────────────────────────
// Collect function attributes that appear after the parameter list closing ')'
// and before the opening '{' (for definitions) or the next top-level construct
// (for declarations).  Returns a vector of attribute strings that can be
// re-emitted verbatim (e.g. "#0", "nounwind", "uwtable",
// "\"frame-pointer\"=\"all\"", etc.)
std::vector<std::string> IRParser::collect_function_attrs() {
    static const std::unordered_set<std::string> top_level_keywords = {
        "define", "declare", "target", "attributes", "module", "source_filename"
    };
    std::vector<std::string> attrs;
    while (peek_token().type != TokenType::Eof) {
        auto tok = peek_token();
        // Stop at function-body open
        if (tok.type == TokenType::Punctuation && tok.text == "{") break;
        // Stop at the next top-level construct (declaration case)
        if (tok.type == TokenType::Keyword && top_level_keywords.count(tok.text)) break;
        if (tok.type == TokenType::Punctuation && tok.text == "@") break;
        if (tok.type == TokenType::Punctuation && tok.text == "%") {
            // %name = type ...  → next named type definition
            if (pos_ + 2 < tokens_.size() &&
                tokens_[pos_ + 1].type == TokenType::Identifier &&
                tokens_[pos_ + 2].type == TokenType::Punctuation &&
                tokens_[pos_ + 2].text == "=") break;
        }
        // Distinguish metadata attachments (!dbg !N) from metadata
        // definitions (!N = ... or !name = ...).  Attachments are consumed
        // by skip_metadata() below; definitions cause us to stop so the
        // outer parse_module loop can dispatch them.
        if (tok.type == TokenType::Punctuation && tok.text == "!") {
            // Lookahead: !<id-or-name> =
            if (pos_ + 3 < tokens_.size() &&
                (tokens_[pos_ + 1].type == TokenType::Number ||
                 tokens_[pos_ + 1].type == TokenType::Identifier) &&
                tokens_[pos_ + 2].type == TokenType::Punctuation &&
                tokens_[pos_ + 2].text == "=") {
                break;
            }
            // Bare attachment — fall through to skip_metadata() below.
        }

        if (tok.type == TokenType::Punctuation && !tok.text.empty() && tok.text[0] == '#') {
            // Attribute group reference: #0, #1, etc.
            attrs.push_back(tok.text);
            next_token();
        } else if (tok.type == TokenType::Keyword) {
            // Named function attribute: nounwind, noinline, uwtable, etc.
            attrs.push_back(tok.text);
            next_token();
            // Handle parenthesized args (e.g. align(N))
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "(") {
                std::string arg = tok.text + "(";
                next_token(); // consume '('
                int depth = 1;
                while (peek_token().type != TokenType::Eof && depth > 0) {
                    auto t = peek_token();
                    if (t.type == TokenType::Punctuation && t.text == "(") depth++;
                    if (t.type == TokenType::Punctuation && t.text == ")") depth--;
                    if (depth <= 0) break;
                    arg += t.text;
                    next_token();
                }
                if (peek_token().type == TokenType::Punctuation && peek_token().text == ")") {
                    arg += ")";
                    next_token();
                }
                // Replace the last pushed attr with the full arg version
                attrs.back() = arg;
            }
            // 'align N' — only 'align' takes a bare number argument
            if (tok.text == "align" && peek_token().type == TokenType::Number) {
                attrs.back() += " " + peek_token().text;
                next_token(); // consume alignment count
            }
        } else if (tok.type == TokenType::String) {
            // Key-value attribute: "frame-pointer"="all"
            std::string kv = tok.text;
            next_token();
            if (peek_token().type == TokenType::Punctuation && peek_token().text == "=") {
                next_token(); // consume '='
                kv += "=";
                auto val = peek_token();
                if (val.type != TokenType::Eof) {
                    kv += val.text;
                    next_token();
                }
            }
            attrs.push_back(kv);
        } else if (tok.type == TokenType::Punctuation && tok.text == "!") {
            // Metadata attachment after function params — skip it
            skip_metadata();
        } else {
            // Unknown token — consume it to avoid infinite loop
            next_token();
        }
    }
    return attrs;
}

// ── token_to_text() ────────────────────────────────────────────────────────
// Render a token back to its textual form.  String tokens already include
// their surrounding quotes (the lexer stores them that way), and the
// `#N` attribute-group token is also stored complete, so for most tokens
// `tok.text` is already the correct rendering.  This helper exists for
// uniformity and future extension (e.g. escaping).
std::string IRParser::token_to_text(const Token& tok) const {
    return tok.text;
}

// ── parse_attribute_group() ────────────────────────────────────────────────
//   attributes #N = { <body> }
//
// We capture <body> verbatim so the emitter can regenerate the line.  The
// body typically contains keywords like `nounwind uwtable "frame-pointer"="all"`
// plus parameter-attribute syntax that clunk's type system doesn't model
// (e.g. `{ ptr nocapture noundef readonly }`).  Storing the raw text avoids
// needing to understand every attribute form.
void IRParser::parse_attribute_group(std::shared_ptr<ir::Module> mod) {
    next_token(); // consume 'attributes'

    // The next token is the attribute-group id: '#0', '#1', etc.
    // The lexer already grouped '#N' into a single Punctuation token.
    auto id_tok = next_token();
    std::string id = id_tok.text;
    if (id.empty() || id[0] != '#') {
        // Malformed — bail out.  Try to recover by skipping to '}'.
        while (peek_token().type != TokenType::Eof &&
               !(peek_token().type == TokenType::Punctuation && peek_token().text == "}")) {
            next_token();
        }
        if (peek_token().type == TokenType::Punctuation && peek_token().text == "}") {
            next_token();
        }
        return;
    }

    if (!(peek_token().type == TokenType::Punctuation && peek_token().text == "=")) {
        return;
    }
    next_token(); // consume '='

    if (!(peek_token().type == TokenType::Punctuation && peek_token().text == "{")) {
        return;
    }
    next_token(); // consume '{'

    // Capture the body verbatim, with sensible spacing between tokens.
    // We walk tokens until the matching '}', reconstructing text.
    std::string body;
    body.reserve(64);
    int depth = 1;  // already inside one '{'

    while (peek_token().type != TokenType::Eof && depth > 0) {
        auto tok = peek_token();
        if (tok.type == TokenType::Punctuation && tok.text == "{") {
            ++depth;
            body += " {";
            next_token();
            continue;
        }
        if (tok.type == TokenType::Punctuation && tok.text == "}") {
            --depth;
            next_token();
            if (depth <= 0) break;
            body += " }";
            continue;
        }
        // Spacing rule: insert a separator space unless:
        //   - body is empty (no leading space)
        //   - last char is already a space or '{' (no double-space)
        //   - last char is '!' (so `!N` and `!"..."` stay atomic — no `! N`)
        //   - current token is ',' or '=' (no leading space before these)
        //   - current token is a String following '=' (so `key="val"` not `key= "val"`)
        if (!body.empty() && body.back() != ' ' && body.back() != '{' &&
            body.back() != '!' &&
            tok.text != "," && tok.text != "=" &&
            !(tok.type == TokenType::String && body.back() == '=')) {
            body += ' ';
        }
        body += token_to_text(tok);
        next_token();
    }

    // Trim leading/trailing whitespace.
    while (!body.empty() && (body.front() == ' ' || body.front() == '\t')) {
        body.erase(body.begin());
    }
    while (!body.empty() && (body.back() == ' ' || body.back() == '\t')) {
        body.pop_back();
    }

    mod->add_attribute_group(id, body);
}

// ── parse_named_metadata() ─────────────────────────────────────────────────
//   !<name> = !{ ... }      (e.g. !llvm.ident = !{!0})
//   !<name> = !N             (e.g. !dbg.cu = !0)
//
// We capture the RHS verbatim.  The name is stored without the leading '!'.
// `!llvm.module.flags` is handled elsewhere (parse_llvm_module_flags) and
// should not reach here — the dispatch in parse_module checks for it first.
void IRParser::parse_named_metadata(std::shared_ptr<ir::Module> mod) {
    next_token(); // consume '!'
    auto name_tok = next_token();
    std::string name = name_tok.text;

    if (!(peek_token().type == TokenType::Punctuation && peek_token().text == "=")) {
        return;
    }
    next_token(); // consume '='

    // Capture the RHS verbatim.  This is typically `!{...}`, `!N`, or
    // `!{!N, !M, ...}`.  We reconstruct from tokens until we hit something
    // that looks like the start of the next top-level construct.
    std::string body;
    body.reserve(32);

    // If the RHS starts with '!', consume the '!' first so the loop below
    // is simpler.
    bool starts_with_bang = (peek_token().type == TokenType::Punctuation &&
                             peek_token().text == "!");
    if (starts_with_bang) {
        body += '!';
        next_token();
    }

    // If after the '!' we have '{', walk to the matching '}'.
    if (peek_token().type == TokenType::Punctuation && peek_token().text == "{") {
        body += "{";
        next_token();
        int depth = 1;
        while (peek_token().type != TokenType::Eof && depth > 0) {
            auto tok = peek_token();
            if (tok.type == TokenType::Punctuation && tok.text == "{") {
                ++depth;
                body += " {";
                next_token();
                continue;
            }
            if (tok.type == TokenType::Punctuation && tok.text == "}") {
                --depth;
                next_token();
                if (depth <= 0) break;
                body += " }";
                continue;
            }
            if (!body.empty() && body.back() != ' ' && body.back() != '{' &&
                body.back() != '!' &&
                tok.text != "," && tok.text != "=") {
                body += ' ';
            }
            body += token_to_text(tok);
            next_token();
        }
        // Close the brace.  clang emits `!{!0}` / `!{!"..."}` / `!{!0, !1}`
        // — NO leading space before `}`.  Just append `}`.
        body += "}";
    } else if (peek_token().type == TokenType::Number) {
        // Simple form: `!name = !N`  (e.g. `!dbg.cu = !0`)
        body += peek_token().text;
        next_token();
    } else if (peek_token().type == TokenType::Identifier) {
        // `!name = !othername`  (rare, but legal)
        body += peek_token().text;
        next_token();
    } else {
        // Unknown RHS — let the dispatcher's fallback handle it.
        return;
    }

    mod->add_named_metadata(name, body);
}

// ── parse_metadata_def() ───────────────────────────────────────────────────
//   !N = !{ ... }            (regular metadata node)
//   !N = distinct !{ ... }   (distinct metadata node)
//   !N = !{!"string", ...}   (typical for !llvm.ident entries)
//
// try_parse_module_flag_entry() has already been attempted for `!N = !{ i32,
// !"key", ... }` patterns and returned false, so we know this is NOT a
// module-flag entry.  Capture it verbatim as a generic metadata def.
void IRParser::parse_metadata_def(std::shared_ptr<ir::Module> mod) {
    // Caller already verified: '!' Number '=' ...
    next_token(); // consume '!'
    auto id_tok = next_token(); // the number
    std::string id = id_tok.text;

    next_token(); // consume '='

    std::string body;
    body.reserve(64);

    // Optional `distinct` keyword.
    if (peek_token().type == TokenType::Identifier && peek_token().text == "distinct") {
        body += "distinct ";
        next_token();
    }

    // RHS starts with '!'
    if (peek_token().type == TokenType::Punctuation && peek_token().text == "!") {
        body += '!';
        next_token();
    } else {
        // Unexpected — bail with what we have so far; the dispatcher's
        // outer loop will pick up at the next token.
        if (!body.empty()) {
            mod->add_metadata_def(id, body);
        }
        return;
    }

    // Body forms: !{ ... }  or  !N  or  !"string"
    if (peek_token().type == TokenType::Punctuation && peek_token().text == "{") {
        body += "{";
        next_token();
        int depth = 1;
        while (peek_token().type != TokenType::Eof && depth > 0) {
            auto tok = peek_token();
            if (tok.type == TokenType::Punctuation && tok.text == "{") {
                ++depth;
                body += " {";
                next_token();
                continue;
            }
            if (tok.type == TokenType::Punctuation && tok.text == "}") {
                --depth;
                next_token();
                if (depth <= 0) break;
                body += " }";
                continue;
            }
            if (!body.empty() && body.back() != ' ' && body.back() != '{' &&
                body.back() != '!' &&
                tok.text != "," && tok.text != "=") {
                body += ' ';
            }
            body += token_to_text(tok);
            next_token();
        }
        // Close the brace.  clang emits `!{!0}` / `!{!"..."}` / `!{!0, !1}`
        // — NO leading space before `}`.  Just append `}`.
        body += "}";
    } else if (peek_token().type == TokenType::Number) {
        body += peek_token().text;
        next_token();
    } else if (peek_token().type == TokenType::String) {
        body += peek_token().text;
        next_token();
    } else if (peek_token().type == TokenType::Identifier) {
        body += peek_token().text;
        next_token();
    }

    mod->add_metadata_def(id, body);
}

// ── parse_module_asm() ─────────────────────────────────────────────────────
//   module asm "..."
//   module asm "...\\n\\t..."
//
// We capture the string (with surrounding quotes) so the emitter can
// regenerate the line.  Multiple `module asm` lines are stored in order.
void IRParser::parse_module_asm(std::shared_ptr<ir::Module> mod) {
    next_token(); // consume 'module'

    if (!(peek_token().type == TokenType::Keyword && peek_token().text == "asm")) {
        // `module ...` that isn't `module asm` — fall back to skipping.
        while (peek_token().type != TokenType::Eof &&
               peek_token().type != TokenType::Keyword &&
               !(peek_token().type == TokenType::Punctuation && peek_token().text == "@")) {
            next_token();
        }
        return;
    }
    next_token(); // consume 'asm'

    // The string token includes its surrounding quotes.
    if (peek_token().type == TokenType::String) {
        mod->add_module_asm(peek_token().text);
        next_token();
    }
}

} // namespace clunk::parser
