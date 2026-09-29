import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import Chunk, HybridRetriever, ScholarshipRAG, clean_page_text, tokenize
from langsmith_evaluate import dataset_name_for, make_evaluator, retrieve_outputs, score_outputs


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("P001-C001", 1, "4.3", "Enrolment load", "Full-time enrolment requires at least 36 credit points per semester."),
            Chunk("P002-C002", 2, "8.131", "WIL Grant", "If you do not undertake your WIL placement, you need to return the grant funds to RMIT."),
            Chunk("P003-C003", 3, "3", "Communication", "Scholarship correspondence is sent to the RMIT student email account."),
        ]
    def test_tokenizer(self):
        self.assertEqual(tokenize("What is the WIL Grant?"), ["wil", "grant"])

    def test_header_cleanup(self):
        cleaned = clean_page_text("Document: Scholarships Terms\nPage 2 of 79\nUseful text")
        self.assertEqual(cleaned, "Useful text")

    def test_bm25_finds_wil(self):
        hit = HybridRetriever(self.chunks).search("return WIL grant funds", method="bm25")[0]
        self.assertEqual(hit["chunk"].section, "8.131")

    def test_unknown_question_is_rejected(self):
        result = ScholarshipRAG(self.chunks).ask("What is the weather on Mars?", use_ollama=False)
        self.assertFalse(result["answerable"])

    def test_generic_leave_routes_to_general_policy(self):
        chunks = self.chunks + [Chunk("P004-C004", 4, "2.6", "Leave of absence", "Not all scholarships allow leave of absence; check the specific conditions.")]
        result = ScholarshipRAG(chunks).ask("What happens if I take leave?", use_ollama=False)
        self.assertTrue(result["answerable"])
        self.assertEqual(result["sources"][0]["section"], "2.6")

    def test_sources_do_not_mix_sections(self):
        result = ScholarshipRAG(self.chunks).ask("What happens to the WIL Grant if I do not undertake placement?", use_ollama=False)
        self.assertEqual({source["section"] for source in result["sources"]}, {"8.131"})

    def test_langsmith_retrieval_metrics(self):
        outputs = retrieve_outputs(HybridRetriever(self.chunks), "return WIL grant funds")
        reference = {"sections": ["8.131"]}
        self.assertEqual(make_evaluator("bm25", "hit_at_1")(outputs, reference)["score"], 1)
        self.assertEqual(make_evaluator("tfidf", "hit_at_3")(outputs, reference)["score"], 1)
        scores = score_outputs(outputs, reference)
        self.assertEqual(scores["bm25_rank"], 1)
        self.assertEqual(scores["tfidf_hit_at_1"], 1)

    def test_langsmith_dataset_name_changes_with_content(self):
        first = dataset_name_for([{"id": "Q1", "question": "One", "sections": ["1"]}])
        second = dataset_name_for([{"id": "Q1", "question": "Two", "sections": ["1"]}])
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
