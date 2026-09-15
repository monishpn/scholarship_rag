import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import Chunk, LexicalRetriever, clean_page_text, tokenize

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        chunks = [
            Chunk("P001-C001",1,"4.3","Enrolment load","Full-time enrolment requires at least 36 credit points per semester."),
            Chunk("P002-C002",2,"8.131","WIL Grant","If you do not undertake your WIL placement, return the grant funds to RMIT."),
            Chunk("P003-C003",3,"3","Communication","Scholarship correspondence is sent to the RMIT student email account."),
        ]
        self.retriever = LexicalRetriever(chunks)
    def test_tokenizer(self): self.assertEqual(tokenize("What is the WIL Grant?"),["wil","grant"])
    def test_cleanup(self): self.assertEqual(clean_page_text("Page 2 of 79\nUseful text"),"Useful text")
    def test_bm25(self): self.assertEqual(self.retriever.compare("return WIL grant funds")["bm25"][0]["section"],"8.131")
    def test_tfidf(self): self.assertEqual(self.retriever.compare("return WIL grant funds")["tfidf"][0]["section"],"8.131")
    def test_top_three(self):
        result=self.retriever.compare("scholarship correspondence")
        self.assertEqual(len(result["bm25"]),3)
        self.assertEqual(len(result["tfidf"]),3)

if __name__ == "__main__": unittest.main()
