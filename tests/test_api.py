"""API route tests for health, readiness, and translation endpoints."""

import unittest

try:
    from fastapi.testclient import TestClient
    from backend.app.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI required for API route tests")
class TestTranslationAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("version", data)

    def test_readiness_check(self):
        response = self.client.get("/api/v1/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_loaded", data)

    def test_model_info(self):
        response = self.client.get("/api/v1/models/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["architecture"], "Encoder-Decoder Attention Network")
        self.assertIn("en", data["source_languages"])

    def test_translation_request_valid(self):
        payload = {
            "source_text": "Machine learning is powerful.",
            "source_lang": "en",
            "target_lang": "hi",
            "decoding_strategy": "beam_search",
            "beam_size": 4,
        }
        response = self.client.post("/api/v1/translations", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("translated_text", data)
        self.assertEqual(data["source_lang"], "en")
        self.assertEqual(data["target_lang"], "hi")

    def test_invalid_language_direction(self):
        payload = {
            "source_text": "Bonjour",
            "source_lang": "fr",
            "target_lang": "es",
        }
        response = self.client.post("/api/v1/translations", json=payload)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
