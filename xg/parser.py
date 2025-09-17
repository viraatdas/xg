from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import ast
from .errors import XGParseError
from .types import Dimension, TensorType, ResultType, get_primitive, dimension_from_token


@dataclass
class Token:
    kind: str
    value: str


KEYWORDS = {
    "def": "DEF",
    "return": "RETURN",
    "if": "IF",
    "else": "ELSE",
    "with": "WITH",
}


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.index = 0

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self.index < self.length:
            ch = self.source[self.index]
            if ch in " \t\r\n":
                self.index += 1
                continue
            if ch == "#":
                self._consume_comment()
                continue
            if ch.isalpha() or ch == "_":
                tokens.append(self._consume_identifier())
                continue
            if ch.isdigit():
                tokens.append(self._consume_number())
                continue
            if ch == ":":
                tokens.append(Token("COLON", ch))
                self.index += 1
                continue
            if ch == ",":
                tokens.append(Token("COMMA", ch))
                self.index += 1
                continue
            if ch == "[":
                tokens.append(Token("LBRACKET", ch))
                self.index += 1
                continue
            if ch == "]":
                tokens.append(Token("RBRACKET", ch))
                self.index += 1
                continue
            if ch == "(":
                tokens.append(Token("LPAREN", ch))
                self.index += 1
                continue
            if ch == ")":
                tokens.append(Token("RPAREN", ch))
                self.index += 1
                continue
            if ch == "{" or ch == "}":
                tokens.append(Token("LBRACE" if ch == "{" else "RBRACE", ch))
                self.index += 1
                continue
            if ch == "-" and self._peek_char() == ">":
                tokens.append(Token("ARROW", "->"))
                self.index += 2
                continue
            if ch == "=" and self._peek_char() == "=":
                tokens.append(Token("EQ", "=="))
                self.index += 2
                continue
            if ch in "+-*/@":
                tokens.append(Token("OP", ch))
                self.index += 1
                continue
            if ch == "=":
                tokens.append(Token("ASSIGN", ch))
                self.index += 1
                continue
            raise XGParseError(f"Unexpected character: {ch}")
        tokens.append(Token("EOF", ""))
        return tokens

    def _peek_char(self) -> str:
        if self.index + 1 < self.length:
            return self.source[self.index + 1]
        return ""

    def _consume_comment(self) -> None:
        while self.index < self.length and self.source[self.index] != "\n":
            self.index += 1

    def _consume_identifier(self) -> Token:
        start = self.index
        while self.index < self.length and (self.source[self.index].isalnum() or self.source[self.index] == "_"):
            self.index += 1
        value = self.source[start:self.index]
        kind = KEYWORDS.get(value, "IDENT")
        return Token(kind, value)

    def _consume_number(self) -> Token:
        start = self.index
        has_dot = False
        while self.index < self.length:
            ch = self.source[self.index]
            if ch == ".":
                if has_dot:
                    raise XGParseError("Malformed number literal")
                has_dot = True
                self.index += 1
                continue
            if not ch.isdigit():
                break
            self.index += 1
        value = self.source[start:self.index]
        return Token("NUMBER", value)


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.index = 0

    def parse(self) -> ast.Program:
        statements: List[ast.Statement] = []
        while not self._check("EOF"):
            statements.append(self._parse_statement())
        return ast.Program(statements=statements)

    def _parse_statement(self) -> ast.Statement:
        if self._match("DEF"):
            return self._parse_function()
        if self._match("RETURN"):
            return self._parse_return()
        if self._match("IF"):
            return self._parse_if()
        if self._match("WITH"):
            return self._parse_with()
        # Assignment or expression statement
        expr = self._parse_expression()
        if isinstance(expr, ast.Identifier) and self._match("ASSIGN"):
            value = self._parse_expression()
            return ast.Assignment(target=expr.name, expression=value)
        if self._match("ASSIGN"):
            raise XGParseError("Invalid assignment target")
        return ast.ExpressionStatement(expression=expr)

    def _parse_function(self) -> ast.FunctionDef:
        name = self._consume("IDENT", "Expected function name").value
        self._consume("LPAREN", "Expected '(' after function name")
        params: List[ast.Parameter] = []
        if not self._check("RPAREN"):
            while True:
                param_type = self._parse_type()
                param_name = self._consume("IDENT", "Expected parameter name").value
                params.append(ast.Parameter(typ=param_type, name=param_name))
                if not self._match("COMMA"):
                    break
        self._consume("RPAREN", "Expected ')' after parameters")
        return_type = None
        if self._match("ARROW"):
            return_type = self._parse_type()
        self._consume("LBRACE", "Expected '{' to start function body")
        body = self._parse_block()
        return ast.FunctionDef(name=name, params=params, return_type=return_type, body=body)

    def _parse_block(self) -> List[ast.Statement]:
        statements: List[ast.Statement] = []
        while not self._check("RBRACE"):
            statements.append(self._parse_statement())
        self._consume("RBRACE", "Expected '}' to close block")
        return statements

    def _parse_return(self) -> ast.Return:
        if self._check("RBRACE"):
            return ast.Return(expression=None)
        expr = self._parse_expression()
        return ast.Return(expression=expr)

    def _parse_if(self) -> ast.If:
        condition = self._parse_expression()
        self._consume("LBRACE", "Expected '{' after if condition")
        then_body = self._parse_block()
        else_body: Optional[List[ast.Statement]] = None
        if self._match("ELSE"):
            self._consume("LBRACE", "Expected '{' after else")
            else_body = self._parse_block()
        return ast.If(condition=condition, then_body=then_body, else_body=else_body)

    def _parse_with(self) -> ast.With:
        contexts: List[ast.WithContext] = []
        contexts.append(ast.WithContext(expression=self._parse_expression()))
        while self._match("COMMA"):
            contexts.append(ast.WithContext(expression=self._parse_expression()))
        if self._match("COLON"):
            pass
        self._consume("LBRACE", "Expected '{' to start with-block body")
        body = self._parse_block()
        return ast.With(contexts=contexts, body=body)

    def _parse_expression(self) -> ast.Expression:
        return self._parse_equality()

    def _parse_equality(self) -> ast.Expression:
        expr = self._parse_additive()
        while self._match("EQ"):
            right = self._parse_additive()
            expr = ast.BinaryOp(left=expr, op="==", right=right)
        return expr

    def _parse_additive(self) -> ast.Expression:
        expr = self._parse_multiplicative()
        while self._match("OP", "+") or self._match("OP", "-"):
            op = self._previous().value
            right = self._parse_multiplicative()
            expr = ast.BinaryOp(left=expr, op=op, right=right)
        return expr

    def _parse_multiplicative(self) -> ast.Expression:
        expr = self._parse_unary()
        while True:
            if self._match("OP", "*") or self._match("OP", "/") or self._match("OP", "@"):
                op = self._previous().value
                right = self._parse_unary()
                expr = ast.BinaryOp(left=expr, op=op, right=right)
                continue
            break
        return expr

    def _parse_unary(self) -> ast.Expression:
        if self._match("OP", "-"):
            right = self._parse_unary()
            zero = ast.Number(value=0)
            return ast.BinaryOp(left=zero, op="-", right=right)
        return self._parse_call()

    def _parse_call(self) -> ast.Expression:
        expr = self._parse_primary()
        while self._match("LPAREN"):
            expr = self._finish_call(expr)
        return expr

    def _finish_call(self, callee: ast.Expression) -> ast.Expression:
        args: List[ast.Expression] = []
        kwargs: List[tuple[str, ast.Expression]] = []
        if not self._check("RPAREN"):
            while True:
                if self._check("IDENT") and self._check_next("ASSIGN"):
                    name = self._consume("IDENT", "Expected keyword name").value
                    self._consume("ASSIGN", "Expected '=' in keyword argument")
                    value = self._parse_expression()
                    kwargs.append((name, value))
                else:
                    args.append(self._parse_expression())
                if not self._match("COMMA"):
                    break
        self._consume("RPAREN", "Expected ')' after arguments")
        return ast.Call(func=callee, args=args, kwargs=kwargs)

    def _parse_primary(self) -> ast.Expression:
        if self._match("NUMBER"):
            token = self._previous()
            if "." in token.value:
                return ast.Number(value=float(token.value))
            return ast.Number(value=int(token.value))
        if self._match("IDENT"):
            return ast.Identifier(name=self._previous().value)
        if self._match("LPAREN"):
            expr = self._parse_expression()
            self._consume("RPAREN", "Expected ')' after expression")
            return expr
        raise XGParseError("Expected expression")

    def _parse_type(self):
        name_token = self._consume("IDENT", "Expected type name")
        name = name_token.value
        if name == "Tensor":
            self._consume("LBRACKET", "Expected '[' after Tensor")
            dtype = self._parse_type()
            self._consume("COMMA", "Expected ',' before Tensor dimensions")
            dims: List[Dimension] = [self._parse_dimension()]
            while self._match("COMMA"):
                dims.append(self._parse_dimension())
            self._consume("RBRACKET", "Expected ']' after Tensor type")
            return TensorType(dtype=dtype, dims=tuple(dims))
        if name == "Result":
            self._consume("LBRACKET", "Expected '[' after Result")
            inner = self._parse_type()
            self._consume("RBRACKET", "Expected ']' after Result type")
            return ResultType(inner=inner)
        return get_primitive(name)

    def _parse_dimension(self) -> Dimension:
        if self._match("NUMBER"):
            token = self._previous()
            return dimension_from_token(int(token.value))
        if self._match("IDENT"):
            return dimension_from_token(self._previous().value)
        raise XGParseError("Expected dimension literal")

    def _match(self, kind: str, value: Optional[str] = None) -> bool:
        if self._check(kind, value):
            self.index += 1
            return True
        return False

    def _consume(self, kind: str, message: str) -> Token:
        if not self._check(kind):
            raise XGParseError(message)
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _check(self, kind: str, value: Optional[str] = None) -> bool:
        if self.index >= len(self.tokens):
            return False
        token = self.tokens[self.index]
        if token.kind != kind:
            return False
        if value is not None and token.value != value:
            return False
        return True

    def _check_next(self, kind: str) -> bool:
        if self.index + 1 >= len(self.tokens):
            return False
        return self.tokens[self.index + 1].kind == kind

    def _previous(self) -> Token:
        return self.tokens[self.index - 1]


def parse_source(source: str) -> ast.Program:
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
