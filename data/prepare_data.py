#!/usr/bin/env python3
"""Generate a deterministic synthetic text-to-SQL train/eval dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """You translate a user's request into one SQL SELECT statement.
Use only these tables and columns:
- employees(id, name, department, title, salary, location)
- projects(id, name, owner_id, budget)
Return only SQL. Do not add markdown or an explanation."""

# Synthetic examples only. These fictional records describe a schema, not people.
TRAIN_EXAMPLES = [
    ("Show every employee record.", "SELECT * FROM employees;"),
    ("List the names of all employees.", "SELECT name FROM employees;"),
    ("Show employee names and job titles.", "SELECT name, title FROM employees;"),
    ("List all employees in Engineering.", "SELECT * FROM employees WHERE department = 'Engineering';"),
    ("Find employees who work in Boston.", "SELECT * FROM employees WHERE location = 'Boston';"),
    ("Show names and salaries for employees earning over 100000.", "SELECT name, salary FROM employees WHERE salary > 100000;"),
    ("Count all employees.", "SELECT COUNT(*) FROM employees;"),
    ("What is the average salary across employees?", "SELECT AVG(salary) FROM employees;"),
    ("Find the average salary in Engineering.", "SELECT AVG(salary) FROM employees WHERE department = 'Engineering';"),
    ("List the departments represented in the employee table.", "SELECT DISTINCT department FROM employees;"),
    ("Count employees in each department.", "SELECT department, COUNT(*) FROM employees GROUP BY department;"),
    ("Show the three employees with the highest salaries.", "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3;"),
    ("List projects with a budget above 250000.", "SELECT name, budget FROM projects WHERE budget > 250000;"),
    ("Calculate the total budget for all projects.", "SELECT SUM(budget) FROM projects;"),
    ("Count the projects.", "SELECT COUNT(*) FROM projects;"),
    ("Show each project with the name of its owner.", "SELECT p.name, e.name FROM projects AS p JOIN employees AS e ON p.owner_id = e.id;"),
    ("List employees ordered by name.", "SELECT name FROM employees ORDER BY name;"),
    ("Find the highest salary.", "SELECT MAX(salary) FROM employees;"),
    ("Find the lowest project budget.", "SELECT MIN(budget) FROM projects;"),
    ("Show employees whose title is Data Scientist.", "SELECT * FROM employees WHERE title = 'Data Scientist';"),
    ("Find employees in the Finance department who work in New York.", "SELECT * FROM employees WHERE department = 'Finance' AND location = 'New York';"),
    ("Show the total salary by department.", "SELECT department, SUM(salary) FROM employees GROUP BY department;"),
    ("List project names from highest budget to lowest budget.", "SELECT name, budget FROM projects ORDER BY budget DESC;"),
    ("Show the names of employees who own a project.", "SELECT DISTINCT e.name FROM employees AS e JOIN projects AS p ON p.owner_id = e.id;"),
]

EVAL_EXAMPLES = [
    ("How many people work in Support?", "SELECT COUNT(*) FROM employees WHERE department = 'Support';"),
    ("What is the mean pay for Design employees?", "SELECT AVG(salary) FROM employees WHERE department = 'Design';"),
    ("Return the five highest-paid employees with their salaries.", "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5;"),
    ("What is the combined budget of projects owned by employees in Engineering?", "SELECT SUM(p.budget) FROM projects AS p JOIN employees AS e ON p.owner_id = e.id WHERE e.department = 'Engineering';"),
    ("Show the number of projects for each owner.", "SELECT owner_id, COUNT(*) FROM projects GROUP BY owner_id;"),
    ("List employee names in Seattle who work in Operations.", "SELECT name FROM employees WHERE location = 'Seattle' AND department = 'Operations';"),
]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_train_rows() -> list[dict[str, object]]:
    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": sql},
            ]
        }
        for question, sql in TRAIN_EXAMPLES
    ]


def build_eval_rows() -> list[dict[str, str]]:
    return [
        {"question": question, "expected_sql": sql}
        for question, sql in EVAL_EXAMPLES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "generated",
        help="Directory for train.jsonl and eval.jsonl.",
    )
    args = parser.parse_args()

    write_jsonl(args.output_dir / "train.jsonl", build_train_rows())
    write_jsonl(args.output_dir / "eval.jsonl", build_eval_rows())
    print(
        f"Wrote {len(TRAIN_EXAMPLES)} training and {len(EVAL_EXAMPLES)} evaluation examples "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
