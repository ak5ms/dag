"""Small typed C++ syntax tree used by formula code generation.

Keeping C++ structure as data makes emission composable and testable without a
template engine.  Rendering is centralized here; operator emitters never build
translation units by incrementally concatenating source strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class CppNode(Protocol):
    def render(self, indent: int = 0) -> str: ...


def _pad(indent: int) -> str:
    return " " * indent


@dataclass(frozen=True)
class Raw:
    text: str

    def render(self, indent: int = 0) -> str:
        return _pad(indent) + self.text


@dataclass(frozen=True)
class Include:
    path: str

    def render(self, indent: int = 0) -> str:
        return f'{_pad(indent)}#include "{self.path}"'


@dataclass(frozen=True)
class Declaration:
    type_name: str
    name: str
    initializer: str | None = None
    qualifiers: str = ""

    def render(self, indent: int = 0) -> str:
        prefix = f"{self.qualifiers} " if self.qualifiers else ""
        init = f" = {self.initializer}" if self.initializer is not None else ""
        return f"{_pad(indent)}{prefix}{self.type_name} {self.name}{init};"


@dataclass(frozen=True)
class Statement:
    expression: str
    comment: str | None = None

    def render(self, indent: int = 0) -> str:
        suffix = f"  // {self.comment}" if self.comment else ""
        return f"{_pad(indent)}{self.expression};{suffix}"


@dataclass(frozen=True)
class Block:
    statements: tuple[CppNode, ...]

    def render(self, indent: int = 0) -> str:
        return "\n".join(statement.render(indent) for statement in self.statements)


@dataclass(frozen=True)
class Function:
    return_type: str
    name: str
    parameters: tuple[str, ...]
    body: Block
    qualifiers: str = "inline"

    def render(self, indent: int = 0) -> str:
        signature = f"{_pad(indent)}{self.qualifiers} {self.return_type} {self.name}({', '.join(self.parameters)})"
        return f"{signature} {{\n{self.body.render(indent + 2)}\n{_pad(indent)}}}"


@dataclass(frozen=True)
class Struct:
    name: str
    members: tuple[CppNode, ...]

    def render(self, indent: int = 0) -> str:
        body = "\n".join(member.render(indent + 2) for member in self.members)
        return f"{_pad(indent)}struct {self.name} {{\n{body}\n{_pad(indent)}}};"


@dataclass(frozen=True)
class Namespace:
    name: str
    declarations: tuple[CppNode, ...]

    def render(self, indent: int = 0) -> str:
        body = "\n\n".join(item.render(indent) for item in self.declarations)
        return f"{_pad(indent)}namespace {self.name} {{\n{body}\n{_pad(indent)}}} // namespace {self.name}"


@dataclass(frozen=True)
class TranslationUnit:
    includes: tuple[Include, ...]
    declarations: tuple[CppNode, ...]
    banner: str

    def render(self, indent: int = 0) -> str:
        del indent
        sections = [f"// {self.banner}", *(include.render() for include in self.includes), "", *(declaration.render() for declaration in self.declarations)]
        return "\n".join(sections) + "\n"
