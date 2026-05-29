from __future__ import annotations

import csv
import io
import re
from statistics import mean, median, multimode
from typing import Any

from services.models import LoadedDocument
from services.response_format import NOT_FOUND, format_table_answer


class TableQA:
    def __init__(self, documents: list[LoadedDocument]):
        self.tables = []
        for document in documents:
            for table in document.metadata.get("tables", []):
                self.tables.append(table)

    @property
    def has_tables(self) -> bool:
        return bool(self.tables)

    def answer(self, question: str) -> str | None:
        if not self.tables:
            return None
        q = question.lower().strip()

        if self._asks_preview(q):
            rows = self._all_rows()[:20]
            if not rows:
                return format_table_answer(NOT_FOUND)
            return format_table_answer(
                self._format_rows(rows),
                f"Showing the first {len(rows)} row(s) from the provided structured file.",
            )

        if self._asks_total_count(q):
            total = sum(int(table.get("row_count", len(table.get("rows", [])))) for table in self.tables)
            return format_table_answer(self._metric_table("Count of records", total), f"The provided dataset contains {total} total records.")

        if self._asks_entity_count(q):
            total = sum(int(table.get("row_count", len(table.get("rows", [])))) for table in self.tables)
            noun = self._entity_label(q)
            return format_table_answer(self._metric_table(f"Count of {noun}", total), f"The file has {total} row(s), so it contains {total} {noun} based on the table records.")

        if "how many" in q and "name" in q:
            values = self._column_values_by_keywords(["name"])
            if values:
                count = len([value for value in values if str(value).strip()])
                return format_table_answer(self._metric_table("Count of names", count), f"There are {count} non-empty name entries in the provided file.")
            return format_table_answer(NOT_FOUND)

        starts = self._starts_with_letter(q)
        if starts:
            column = self._best_column(["name", "employee", "employee name", "full name"])
            if not column:
                return format_table_answer(NOT_FOUND)
            rows = self._filter_rows_startswith(column, starts)
            if not rows:
                return format_table_answer(NOT_FOUND)
            return format_table_answer(self._format_rows(rows), f"The file contains {len(rows)} matching row(s) where {column} begins with {starts.upper()}.")

        detail_name = self._details_name(question)
        if detail_name:
            column = self._best_column(["name", "employee", "employee name", "full name"])
            if not column:
                return format_table_answer(NOT_FOUND)
            rows = self._filter_rows_equals(column, detail_name)
            if not rows:
                return format_table_answer(NOT_FOUND)
            return format_table_answer(self._format_rows(rows), f"The provided file contains {len(rows)} matching record(s) for {detail_name}.")

        comparison = self._comparison(q)
        if comparison:
            column = self._mentioned_column(q) or self._best_numeric_column(["salary", "marks", "age", "score", "amount", "price"])
            if not column:
                return format_table_answer(NOT_FOUND)
            rows = self._filter_numeric(column, comparison[0], comparison[1])
            if not rows:
                return format_table_answer(NOT_FOUND)
            op_text = "above" if comparison[0] == "gt" else "below"
            return format_table_answer(self._format_rows(rows), f"The provided file contains {len(rows)} row(s) where {column} is {op_text} {self._format_number(comparison[1])}.")

        grouped = self._grouped_average_request(q)
        if grouped:
            group_col, value_col = grouped
            return self._highest_group_average(group_col, value_col)

        sort_request = self._sort_request(q)
        if sort_request:
            column, descending, limit = sort_request
            rows = self._sort_rows(column, descending)
            if not rows:
                return format_table_answer(NOT_FOUND)
            direction = "descending" if descending else "ascending"
            shown = rows[:limit]
            return format_table_answer(self._format_rows(shown), f"The result shows the top {len(shown)} row(s), sorted by {column} in {direction} order from the provided file.")

        duplicate_col = self._duplicates_request(q)
        if duplicate_col:
            duplicates = self._duplicates(duplicate_col)
            if not duplicates:
                return format_table_answer(NOT_FOUND)
            rows = [{duplicate_col: value, "Count": count} for value, count in duplicates]
            return format_table_answer(self._format_rows(rows), f"The listed {duplicate_col} values appear more than once in the provided file.")

        contains = self._contains_request(question)
        if contains:
            column, value = contains
            rows = self._filter_rows_contains(column, value)
            if not rows:
                return format_table_answer(NOT_FOUND)
            return format_table_answer(self._format_rows(rows), f"The file contains {len(rows)} matching row(s) where {column} includes '{value}'.")

        if self._asks_missing(q):
            rows = self._missing_values()
            return format_table_answer(self._format_rows(rows), "This table shows missing-value counts by column.")

        agg = self._aggregation(q)
        if agg:
            column = self._mentioned_column(q) or self._best_numeric_column(["salary", "marks", "age", "score", "amount", "price"])
            if not column:
                return format_table_answer(NOT_FOUND)
            return self._aggregate(column, agg)

        mentioned_column = self._mentioned_column(q)
        if mentioned_column and any(word in q for word in ["show", "list", "all", "values"]):
            values = self._column_values(mentioned_column)
            if values:
                unique = self._dedupe([str(value) for value in values if str(value).strip()])
                rows = [{mentioned_column: value} for value in unique[:50]]
                return format_table_answer(self._format_rows(rows), f"The provided file contains {len(unique)} unique value(s) for {mentioned_column}.")

        return None

    def suggestions(self) -> list[str]:
        questions = []
        for table in self.tables[:2]:
            label = table.get("label", "table")
            columns = table.get("columns", [])
            numeric = [column for column in columns if self._is_numeric_column(column)]
            name_col = self._first_matching_column(columns, ["name", "employee", "full name"])
            questions.append(f"How many records are in {label}?")
            questions.append(f"Show the first 10 rows in {label}.")
            questions.append(f"Show missing values summary for {label}.")
            if name_col:
                questions.append(f"How many names are in {label}?")
                questions.append(f"Show all rows where {name_col} starts with A.")
                first_name = self._first_non_empty_value(name_col)
                if first_name:
                    questions.append(f"Show full details of {first_name}.")
                questions.append(f"Are there duplicate {name_col} values?")
            for column in numeric[:3]:
                questions.append(f"What is the highest {column} in {label}?")
                questions.append(f"What is the average {column} in {label}?")
                questions.append(f"What is the median {column} in {label}?")
                questions.append(f"Show rows where {column} is above 50000.")
                questions.append(f"Sort records by {column} from highest to lowest.")
                group_col = self._best_group_column(columns, column)
                if group_col:
                    questions.append(f"Which {group_col} has the highest average {column}?")
            for column in columns[:4]:
                questions.append(f"List all unique {column} values in {label}.")
        return self._dedupe(questions)[:10]

    def artifacts(self, question: str) -> dict[str, str | None]:
        if not self.tables:
            return {"sql_query": None, "excel_formula": None, "csv_data": None}
        q = question.lower().strip()
        table_name = self._table_name()
        columns = self._all_columns()
        numeric = self._mentioned_column(q) or self._best_numeric_column(["salary", "marks", "age", "score", "amount", "price"])
        group_col = self._best_group_column(columns, numeric) if numeric else None
        sql = f"SELECT * FROM {table_name} LIMIT 20;"
        formula = ""

        comparison = self._comparison(q)
        if comparison and numeric:
            operator = ">" if comparison[0] == "gt" else "<"
            sql = f"SELECT * FROM {table_name} WHERE {self._sql_col(numeric)} {operator} {self._format_number(comparison[1])};"
            formula = f'=FILTER(A:Z,{self._excel_col(numeric)}:{self._excel_col(numeric)}{operator}{self._format_number(comparison[1])})'
        elif self._aggregation(q) and numeric:
            agg = self._aggregation(q)
            sql_func = {"max": "MAX", "min": "MIN", "avg": "AVG", "median": "MEDIAN", "mode": "MODE", "sum": "SUM"}.get(agg, "AVG")
            sql = f"SELECT {sql_func}({self._sql_col(numeric)}) AS {agg}_{self._safe_name(numeric)} FROM {table_name};"
            excel_func = {"max": "MAX", "min": "MIN", "avg": "AVERAGE", "median": "MEDIAN", "mode": "MODE.SNGL", "sum": "SUM"}.get(agg, "AVERAGE")
            formula = f"={excel_func}({self._excel_col(numeric)}:{self._excel_col(numeric)})"
        elif self._grouped_average_request(q) and numeric and group_col:
            sql = (
                f"SELECT {self._sql_col(group_col)}, AVG({self._sql_col(numeric)}) AS avg_{self._safe_name(numeric)} "
                f"FROM {table_name} GROUP BY {self._sql_col(group_col)} "
                f"ORDER BY avg_{self._safe_name(numeric)} DESC LIMIT 1;"
            )
            formula = f'=SORTBY(UNIQUE({self._excel_col(group_col)}:{self._excel_col(group_col)}),AVERAGEIF({self._excel_col(group_col)}:{self._excel_col(group_col)},UNIQUE({self._excel_col(group_col)}:{self._excel_col(group_col)}),{self._excel_col(numeric)}:{self._excel_col(numeric)}),-1)'
        elif self._sort_request(q):
            column, descending, limit = self._sort_request(q)
            direction = "DESC" if descending else "ASC"
            sql = f"SELECT * FROM {table_name} ORDER BY {self._sql_col(column)} {direction} LIMIT {limit};"
            formula = f"=SORT(A:Z,{self._column_index(column)},{'-1' if descending else '1'})"
        elif self._duplicates_request(q):
            column = self._duplicates_request(q)
            sql = (
                f"SELECT {self._sql_col(column)}, COUNT(*) AS duplicate_count FROM {table_name} "
                f"GROUP BY {self._sql_col(column)} HAVING COUNT(*) > 1;"
            )
            formula = f'=FILTER(A:Z,COUNTIF({self._excel_col(column)}:{self._excel_col(column)},{self._excel_col(column)}:{self._excel_col(column)})>1)'
        elif self._asks_total_count(q) or self._asks_entity_count(q):
            sql = f"SELECT COUNT(*) AS total_records FROM {table_name};"
            formula = "=COUNTA(A:A)-1"

        return {
            "sql_query": sql,
            "excel_formula": formula or "Use Excel filters or PivotTable based on the displayed result.",
            "csv_data": self._csv_data(),
        }

    def _column_values_by_keywords(self, keywords: list[str]) -> list[Any]:
        column = self._best_column(keywords)
        return self._column_values(column) if column else []

    def _column_values(self, column: str) -> list[Any]:
        values = []
        for table in self.tables:
            for row in table.get("rows", []):
                if column in row:
                    values.append(row[column])
        return values

    def _all_rows(self) -> list[dict]:
        return [row for table in self.tables for row in table.get("rows", [])]

    def _filter_rows_startswith(self, column: str, letter: str) -> list[dict]:
        rows = []
        for table in self.tables:
            for row in table.get("rows", []):
                value = str(row.get(column, "")).strip()
                if value.lower().startswith(letter.lower()):
                    rows.append(row)
        return rows

    def _filter_rows_equals(self, column: str, value: str) -> list[dict]:
        rows = []
        for table in self.tables:
            for row in table.get("rows", []):
                if str(row.get(column, "")).strip().lower() == value.strip().lower():
                    rows.append(row)
        return rows

    def _filter_rows_contains(self, column: str, value: str) -> list[dict]:
        rows = []
        for table in self.tables:
            for row in table.get("rows", []):
                if value.lower() in str(row.get(column, "")).lower():
                    rows.append(row)
        return rows

    def _filter_numeric(self, column: str, operator: str, threshold: float) -> list[dict]:
        rows = []
        for table in self.tables:
            for row in table.get("rows", []):
                value = self._to_float(row.get(column))
                if value is None:
                    continue
                if operator == "gt" and value > threshold:
                    rows.append(row)
                if operator == "lt" and value < threshold:
                    rows.append(row)
        return rows

    def _sort_rows(self, column: str, descending: bool) -> list[dict]:
        rows = [row for table in self.tables for row in table.get("rows", []) if column in row]
        if not rows:
            return []
        numeric_values = [self._to_float(row.get(column)) for row in rows]
        if any(value is not None for value in numeric_values):
            return sorted(rows, key=lambda row: self._to_float(row.get(column)) if self._to_float(row.get(column)) is not None else float("-inf"), reverse=descending)
        return sorted(rows, key=lambda row: str(row.get(column, "")).lower(), reverse=descending)

    def _duplicates(self, column: str) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for value in self._column_values(column):
            text = str(value).strip()
            if text:
                counts[text] = counts.get(text, 0) + 1
        return [(value, count) for value, count in counts.items() if count > 1]

    def _aggregate(self, column: str, agg: str) -> str:
        values = []
        rows_by_value = []
        for table in self.tables:
            for row in table.get("rows", []):
                numeric = self._to_float(row.get(column))
                if numeric is not None:
                    values.append(numeric)
                    rows_by_value.append((numeric, row))
        if not values:
            return format_table_answer(NOT_FOUND)
        if agg == "max":
            value, row = max(rows_by_value, key=lambda item: item[0])
            metric = {"Metric": f"Highest {column}", "Value": self._format_number(value)}
            return format_table_answer(self._format_rows([metric, row]), f"The highest {column} in the provided file is {self._format_number(value)}, and the matching record is shown in the table.")
        if agg == "min":
            value, row = min(rows_by_value, key=lambda item: item[0])
            metric = {"Metric": f"Lowest {column}", "Value": self._format_number(value)}
            return format_table_answer(self._format_rows([metric, row]), f"The lowest {column} in the provided file is {self._format_number(value)}, and the matching record is shown in the table.")
        if agg == "avg":
            value = self._format_number(mean(values))
            return format_table_answer(self._metric_table(f"Average {column}", value), f"The average {column} computed from the provided file is {value}.")
        if agg == "median":
            value = self._format_number(median(values))
            return format_table_answer(self._metric_table(f"Median {column}", value), f"The median {column} computed from the provided file is {value}.")
        if agg == "mode":
            modes = [self._format_number(value) for value in multimode(values)]
            return format_table_answer(self._metric_table(f"Mode {column}", ", ".join(modes)), f"The most frequent {column} value(s) in the provided file are {', '.join(modes)}.")
        if agg == "sum":
            value = self._format_number(sum(values))
            return format_table_answer(self._metric_table(f"Total {column}", value), f"The total {column} computed from the provided file is {value}.")
        return format_table_answer(NOT_FOUND)

    def _highest_group_average(self, group_col: str, value_col: str) -> str:
        groups: dict[str, list[float]] = {}
        for table in self.tables:
            for row in table.get("rows", []):
                group = str(row.get(group_col, "")).strip()
                value = self._to_float(row.get(value_col))
                if group and value is not None:
                    groups.setdefault(group, []).append(value)
        if not groups:
            return format_table_answer(NOT_FOUND)
        averages = {group: mean(values) for group, values in groups.items()}
        best_group, best_value = max(averages.items(), key=lambda item: item[1])
        rows = [
            {group_col: group, f"Average {value_col}": self._format_number(value)}
            for group, value in sorted(averages.items(), key=lambda item: item[1], reverse=True)
        ]
        return format_table_answer(self._format_rows(rows), f"{best_group} has the highest average {value_col} in the provided file, with an average of {self._format_number(best_value)}.")

    def _best_column(self, keywords: list[str]) -> str | None:
        all_columns = [column for table in self.tables for column in table.get("columns", [])]
        return self._first_matching_column(all_columns, keywords)

    def _mentioned_column(self, question_lower: str) -> str | None:
        all_columns = [column for table in self.tables for column in table.get("columns", [])]
        for column in all_columns:
            normalized = str(column).lower().replace("_", " ")
            if normalized in question_lower or normalized.rstrip("s") in question_lower:
                return column
        semantic_keywords = self._semantic_column_keywords(question_lower)
        if semantic_keywords:
            return self._best_column(semantic_keywords)
        return None

    def _sort_request(self, question_lower: str) -> tuple[str, bool, int] | None:
        ranking_words = ["sort", "order", "rank", "top", "bottom", "best", "performer"]
        if not any(word in question_lower for word in ranking_words):
            return None
        column = self._mentioned_column(question_lower) or self._best_numeric_column(
            ["score", "marks", "grade", "salary", "sales", "revenue", "profit", "amount", "price"]
        )
        if not column:
            return None
        descending = any(word in question_lower for word in ["highest", "descending", "top", "largest", "maximum", "best"])
        if any(phrase in question_lower for phrase in ["lowest", "minimum", "ascending", "lowest to highest", "smallest"]):
            descending = False
        if "highest to lowest" in question_lower or "largest to smallest" in question_lower:
            descending = True
        return column, descending, self._limit_request(question_lower)

    def _duplicates_request(self, question_lower: str) -> str | None:
        if "duplicate" not in question_lower and "repeated" not in question_lower:
            return None
        return self._mentioned_column(question_lower) or self._best_column(["name", "employee", "id"])

    def _contains_request(self, question: str) -> tuple[str, str] | None:
        q = question.lower()
        if "contains" not in q and "include" not in q:
            return None
        column = self._mentioned_column(q) or self._best_column(["name", "employee", "department", "category", "title"])
        match = re.search(r"(?:contains|includes?)\s+['\"]?([^'\"]+?)['\"]?(?:\?|$)", question, flags=re.IGNORECASE)
        if column and match:
            value = match.group(1).strip().strip(".?")
            return column, value
        return None

    def _grouped_average_request(self, question_lower: str) -> tuple[str, str] | None:
        if "highest average" not in question_lower and "largest average" not in question_lower:
            return None
        numeric = self._best_numeric_column(["salary", "marks", "score", "amount", "price"])
        all_columns = [column for table in self.tables for column in table.get("columns", [])]
        group = None
        for column in all_columns:
            normalized = str(column).lower().replace("_", " ")
            if normalized in question_lower and not self._is_numeric_column(column):
                group = column
                break
        if not group:
            group = next((column for column in all_columns if not self._is_numeric_column(column)), None)
        if group and numeric:
            return group, numeric
        return None

    def _best_numeric_column(self, preferred: list[str]) -> str | None:
        column = self._best_column(preferred)
        if column and self._is_numeric_column(column):
            return column
        for table in self.tables:
            for candidate in table.get("columns", []):
                if self._is_numeric_column(candidate):
                    return candidate
        return column

    @staticmethod
    def _semantic_column_keywords(question_lower: str) -> list[str]:
        groups = {
            ("score", "scored", "performer", "performance", "mark", "marks", "grade"): ["score", "marks", "grade", "points", "result"],
            ("pay", "salary", "compensation", "earn", "earns", "income"): ["salary", "pay", "income", "compensation"],
            ("sale", "sales", "revenue", "profit", "amount", "price"): ["sales", "revenue", "profit", "amount", "price", "total"],
            ("name", "student", "employee", "person", "customer"): ["name", "student", "employee", "customer", "person"],
            ("date", "month", "year", "time"): ["date", "month", "year", "time"],
        }
        keywords: list[str] = []
        for triggers, columns in groups.items():
            if any(trigger in question_lower for trigger in triggers):
                keywords.extend(columns)
        return keywords

    def _best_group_column(self, columns: list[str], value_column: str) -> str | None:
        preferred = ["department", "team", "category", "location", "grade", "role", "title"]
        for keyword in preferred:
            for column in columns:
                if column != value_column and keyword in str(column).lower() and not self._is_numeric_column(column):
                    return column
        return next(
            (
                column for column in columns
                if column != value_column
                and not self._is_numeric_column(column)
                and "name" not in str(column).lower()
            ),
            None,
        )

    def _first_non_empty_value(self, column: str) -> str | None:
        for value in self._column_values(column):
            text = str(value).strip()
            if text:
                return text
        return None

    def _is_numeric_column(self, column: str) -> bool:
        values = self._column_values(column)
        non_empty = [value for value in values if str(value).strip()]
        if not non_empty:
            return False
        numeric = [value for value in non_empty if self._to_float(value) is not None]
        return len(numeric) / len(non_empty) >= 0.7

    def _missing_values(self) -> list[dict]:
        rows = []
        all_columns = self._all_columns()
        all_rows = self._all_rows()
        for column in all_columns:
            count = sum(1 for row in all_rows if not str(row.get(column, "")).strip())
            rows.append({"Column": column, "Missing Values": count})
        return rows

    def _all_columns(self) -> list[str]:
        columns = []
        for table in self.tables:
            for column in table.get("columns", []):
                if column not in columns:
                    columns.append(column)
        return columns

    def _csv_data(self) -> str:
        rows = self._all_rows()
        columns = self._all_columns()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        return output.getvalue()

    def _table_name(self) -> str:
        label = self.tables[0].get("label", "uploaded_table") if self.tables else "uploaded_table"
        return self._safe_name(label)

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_").lower()
        return cleaned or "uploaded_table"

    def _sql_col(self, column: str) -> str:
        return f'"{column}"'

    def _excel_col(self, column: str) -> str:
        index = self._column_index(column)
        letters = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            letters = chr(65 + remainder) + letters
        return letters or "A"

    def _column_index(self, column: str) -> int:
        columns = self._all_columns()
        return columns.index(column) + 1 if column in columns else 1

    @staticmethod
    def _first_matching_column(columns: list[str], keywords: list[str]) -> str | None:
        lowered = [(column, str(column).lower().replace("_", " ")) for column in columns]
        for keyword in keywords:
            for original, normalized in lowered:
                if keyword in normalized:
                    return original
        return None

    @staticmethod
    def _asks_total_count(question_lower: str) -> bool:
        return any(phrase in question_lower for phrase in ["how many rows", "how many records", "total rows", "total records"])

    @staticmethod
    def _asks_preview(question_lower: str) -> bool:
        return any(phrase in question_lower for phrase in ["show data", "show table", "preview", "first 5", "first five", "first 10", "first ten"])

    @staticmethod
    def _asks_missing(question_lower: str) -> bool:
        return any(phrase in question_lower for phrase in ["missing values", "null values", "empty values", "missing data"])

    @staticmethod
    def _asks_entity_count(question_lower: str) -> bool:
        return "how many" in question_lower and any(
            word in question_lower
            for word in [
                "employee", "employees", "student", "students", "customer", "customers",
                "user", "users", "product", "products", "entry", "entries", "item", "items",
            ]
        )

    @staticmethod
    def _entity_label(question_lower: str) -> str:
        if "employee" in question_lower:
            return "employees"
        if "student" in question_lower:
            return "students"
        if "customer" in question_lower:
            return "customers"
        if "user" in question_lower:
            return "users"
        if "product" in question_lower:
            return "products"
        if "entry" in question_lower or "entries" in question_lower:
            return "entries"
        if "item" in question_lower:
            return "items"
        return "records"

    @staticmethod
    def _starts_with_letter(question_lower: str) -> str | None:
        patterns = [
            r"starting with ['\"]?([a-z])['\"]?",
            r"starts with ['\"]?([a-z])['\"]?",
            r"beginning with ['\"]?([a-z])['\"]?",
            r"names?\s+starting\s+with\s+['\"]?([a-z])['\"]?",
        ]
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _comparison(question_lower: str) -> tuple[str, float] | None:
        patterns = [
            (r"(?:above|more than|greater than|over)\s+([\d,]+(?:\.\d+)?\s*(?:k|lakh|lakhs)?)", "gt"),
            (r"(?:below|less than|under)\s+([\d,]+(?:\.\d+)?\s*(?:k|lakh|lakhs)?)", "lt"),
        ]
        for pattern, operator in patterns:
            match = re.search(pattern, question_lower)
            if match:
                parsed = TableQA._to_float(match.group(1))
                if parsed is not None:
                    return operator, parsed
        return None

    @staticmethod
    def _details_name(question: str) -> str | None:
        match = re.search(r"(?:details of|detail of|details for|record for)\s+([A-Za-z][A-Za-z ]{1,60})", question, flags=re.IGNORECASE)
        if not match:
            return None
        value = match.group(1).strip().strip("?.,")
        stop = {"the", "employee", "person", "record"}
        words = [word for word in value.split() if word.lower() not in stop]
        return " ".join(words) if words else None

    @staticmethod
    def _aggregation(question_lower: str) -> str | None:
        if any(word in question_lower for word in ["highest", "maximum", "max", "largest", "top"]):
            return "max"
        if any(word in question_lower for word in ["lowest", "minimum", "min", "smallest"]):
            return "min"
        if any(word in question_lower for word in ["average", "avg", "mean"]):
            return "avg"
        if "median" in question_lower:
            return "median"
        if "mode" in question_lower or "most frequent" in question_lower:
            return "mode"
        if any(word in question_lower for word in ["sum", "total"]):
            return "sum"
        return None

    @staticmethod
    def _limit_request(question_lower: str) -> int:
        match = re.search(r"\b(?:top|bottom|first|last)\s+(\d{1,3})\b", question_lower)
        if match:
            return max(1, min(int(match.group(1)), 50))
        if "top" in question_lower or "bottom" in question_lower or "best" in question_lower:
            return 5
        return 20

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).replace(",", "").replace("$", "").strip()
        lowered = text.lower()
        multiplier = 1.0
        if lowered.endswith("k"):
            multiplier = 1000.0
            text = text[:-1]
        elif lowered.endswith("lakh") or lowered.endswith("lakhs"):
            multiplier = 100000.0
            text = re.sub(r"lakhs?$", "", lowered).strip()
        if not text:
            return None
        try:
            return float(text) * multiplier
        except ValueError:
            return None

    def _metric_table(self, metric: str, value: Any) -> str:
        return self._format_rows([{"Metric": metric, "Value": value}])

    @staticmethod
    def _format_rows(rows: list[dict]) -> str:
        if not rows:
            return NOT_FOUND
        columns = []
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows[:50]:
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        return "\n".join(lines)

    @staticmethod
    def _format_number(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:.2f}"

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen = []
        for value in values:
            if value and value not in seen:
                seen.append(value)
        return seen
