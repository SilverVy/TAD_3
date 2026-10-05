import unittest

from scripts.download_gbif_butterflies import _qualified_media


class GbifDownloaderTests(unittest.TestCase):
    def test_accepts_cc_by_media_and_keeps_attribution(self):
        record = {
            "extensions": {
                "http://rs.tdwg.org/dwc/terms/Multimedia": [
                    {
                        "http://purl.org/dc/terms/license": (
                            "https://creativecommons.org/licenses/by/4.0/"
                        ),
                        "http://purl.org/dc/terms/identifier": (
                            "https://example.org/butterfly.jpg"
                        ),
                        "http://purl.org/dc/terms/creator": "Photo creator",
                        "http://purl.org/dc/terms/rightsHolder": "Rights holder",
                        "http://purl.org/dc/terms/format": "image/jpeg",
                    }
                ]
            }
        }

        media = _qualified_media(record)

        self.assertIsNotNone(media)
        self.assertEqual(media["license_name"], "CC BY 4.0")
        self.assertEqual(media["creator"], "Photo creator")
        self.assertEqual(media["rights_holder"], "Rights holder")

    def test_rejects_unapproved_license(self):
        record = {
            "extensions": {
                "http://rs.tdwg.org/dwc/terms/Multimedia": [
                    {
                        "http://purl.org/dc/terms/license": (
                            "https://creativecommons.org/licenses/by-nc/4.0/"
                        ),
                        "http://purl.org/dc/terms/identifier": (
                            "https://example.org/butterfly.jpg"
                        ),
                        "http://purl.org/dc/terms/format": "image/jpeg",
                    }
                ]
            }
        }

        self.assertIsNone(_qualified_media(record))


if __name__ == "__main__":
    unittest.main()
