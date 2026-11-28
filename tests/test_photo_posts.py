"""Upload decoding, taxonomy, and factual boundary checks without trained vision weights."""

from io import BytesIO
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from photo_posts import write_photo_post
from scratch.photo_tags import LABELS, decode_photo
from server import WorkspaceHandler


class PhotoPostTests(unittest.TestCase):
    def test_http_accepts_photo_binary_and_validates_confirmed_text(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), WorkspaceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        try:
            stream = BytesIO()
            Image.new('RGB', (64, 64)).save(stream, format='PNG')
            content = stream.getvalue()
            with patch('server.recognize_photo', return_value={'predictions': []}) as recognize:
                connection.request('POST', '/api/photo-tags', content, {'Content-Type': 'image/png'})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                recognize.assert_called_once_with(content)
            connection.request('POST', '/api/photo-tags', b'bad image', {'Content-Type': 'image/png'})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
            connection.request('POST', '/api/photo-post', json.dumps({'labels':['person'], 'confirmed':False}), {'Content-Type':'application/json'})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_full_taxonomy_has_80_unique_labels(self):
        self.assertEqual(len(LABELS), 80)
        self.assertEqual(len(set(LABELS)), 80)
        self.assertIn('pizza', LABELS)
        self.assertIn('cell phone', LABELS)
        self.assertNotIn('barbecue', LABELS)

    def test_image_preserves_composition_with_gray_padding(self):
        stream = BytesIO()
        Image.new('RGB', (160, 80), (255, 0, 0)).save(stream, format='PNG')
        image = decode_photo(stream.getvalue())
        self.assertEqual(image.shape, (3, 64, 64))
        np.testing.assert_array_equal(image[:, 0, 0], [127, 127, 127])
        np.testing.assert_array_equal(image[:, 32, 32], [255, 0, 0])

    def test_transparency_is_composited_onto_gray(self):
        stream = BytesIO()
        Image.new('RGBA', (64, 64), (0, 0, 0, 0)).save(stream, format='PNG')
        self.assertTrue((decode_photo(stream.getvalue()) == 127).all())

    def test_rejects_invalid_tiny_and_oversized_upload(self):
        for content in (b'not an image', b'', b'x' * (8 * 1024 * 1024 + 1)):
            with self.assertRaises(ValueError):
                decode_photo(content)
        stream = BytesIO()
        Image.new('RGB', (8, 8)).save(stream, format='PNG')
        with self.assertRaises(ValueError):
            decode_photo(stream.getvalue())

    def test_writing_requires_confirmed_known_labels(self):
        with patch('photo_posts.rewrite_post', return_value={'versions': ['a', 'b', 'c']}) as writer:
            for labels, confirmed in ((['person'], False), (['barbecue'], True), ([], True)):
                with self.assertRaises(ValueError):
                    write_photo_post(labels, '', 'concise', confirmed)
            writer.assert_not_called()
            result = write_photo_post(['person', 'pizza'], '和室友聚餐', 'concise', True)
            text = writer.call_args.args[0]
            self.assertIn('人物', text)
            self.assertIn('披萨', text)
            self.assertIn('和室友聚餐', text)
            self.assertNotIn('image', text)
            self.assertEqual(result['confirmed_labels'], ['person', 'pizza'])
            self.assertTrue(writer.call_args.kwargs['photo'])


if __name__ == '__main__':
    unittest.main()
