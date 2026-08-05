from dataclasses import FrozenInstanceError
import unittest

from yt_transformer.notation import Tableau, format_notation, parse_notation


class TableauTests(unittest.TestCase):
    def test_constructor_normalizes_rows_to_immutable_tuples(self) -> None:
        source = [[2, 3, 5], [1, 4]]
        tableau = Tableau(source)
        source[0][0] = 50

        self.assertEqual(tableau.rows, ((2, 3, 5), (1, 4)))
        self.assertIsInstance(tableau.rows, tuple)
        self.assertIsInstance(tableau.rows[0], tuple)
        with self.assertRaises(FrozenInstanceError):
            tableau.rows = ()  # type: ignore[misc]

    def test_shape_and_value_validation(self) -> None:
        invalid_rows = (
            ((1,), (2, 3)),
            ((1,), ()),
            ((0,),),
            ((51,),),
            ((True,),),
            ((1.0,),),
        )
        for rows in invalid_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                Tableau(rows)

    def test_transpose_is_an_involution_for_ragged_rows(self) -> None:
        tableau = Tableau(((2, 3, 5), (1, 4)))
        self.assertEqual(tableau.transpose().rows, ((2, 1), (3, 4), (5,)))
        self.assertEqual(tableau.transpose().transpose(), tableau)


class NotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tableau = Tableau(((2, 3, 5), (1, 4)))

    def test_format_all_three_surface_forms(self) -> None:
        self.assertEqual(
            format_notation(self.tableau, "raw"),
            "[YT start] 2 3 5 | 1 4 [YT end]",
        )
        self.assertEqual(
            format_notation(self.tableau, "row"),
            "[YT row start] 2 3 5 | 1 4 [YT row end]",
        )
        self.assertEqual(
            format_notation(self.tableau, "col"),
            "[YT col start] 2 1 | 3 4 | 5 [YT col end]",
        )

    def test_parse_all_three_forms(self) -> None:
        examples = {
            "raw": "[YT start] 2 3 5 | 1 4 [YT end]",
            "row": "[YT row start] 2 3 5 | 1 4 [YT row end]",
            "col": "[YT col start] 2 1 | 3 4 | 5 [YT col end]",
        }
        for expected_kind, text in examples.items():
            with self.subTest(kind=expected_kind):
                tableau, kind = parse_notation(text)
                self.assertEqual(tableau, self.tableau)
                self.assertEqual(kind, expected_kind)
                self.assertEqual(format_notation(tableau, kind), text)

    def test_empty_tableau_has_canonical_round_trip(self) -> None:
        tableau = Tableau(())
        for kind in ("raw", "row", "col"):
            with self.subTest(kind=kind):
                text = format_notation(tableau, kind)  # type: ignore[arg-type]
                reparsed, reparsed_kind = parse_notation(text)
                self.assertEqual(reparsed, tableau)
                self.assertEqual(reparsed_kind, kind)

    def test_rejects_non_canonical_or_invalid_text(self) -> None:
        invalid = (
            "[YT start] 2  3 [YT end]",
            " [YT start] 2 3 [YT end]",
            "[YT start] 02 3 [YT end]",
            "[YT start] 2 | 3 4 [YT end]",  # increasing row lengths
            "[YT col start] 2 | 3 4 [YT col end]",  # invalid column shape
            "[YT start] 0 [YT end]",
            "[YT start] 51 [YT end]",
            "[YT start] 2 3 [YT row end]",
            "[unknown] 2 [unknown end]",
            "[YT start] 2 | | 1 [YT end]",
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_notation(text)

    def test_rejects_unknown_kind_and_wrong_input_type(self) -> None:
        with self.assertRaises(ValueError):
            format_notation(self.tableau, "diagonal")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            format_notation(((1,),), "raw")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            parse_notation(123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
