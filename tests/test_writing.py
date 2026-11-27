"""Writing response validation and lexical retrieval checks; no paid API calls."""

import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import threading
import unittest
from unittest.mock import patch

from models import parse_rewrites
from references import find_references
from server import WorkspaceHandler


class WritingTests(unittest.TestCase):
    def test_parse_requires_three_distinct_bounded_versions(self):
        self.assertEqual(parse_rewrites('prefix {"versions":["one","two","three"]}'), ['one', 'two', 'three'])
        self.assertEqual(parse_rewrites('{"versions":[{"one"},{"two"},{"three"}]}'), ['one', 'two', 'three'])
        for versions in ([], ['a', 'a', 'b'], ['a', 1, 'b'], ['a', 'b', 'x' * 1001]):
            with self.assertRaises(ValueError):
                parse_rewrites(json.dumps({'versions': versions}))

    def test_retrieval_finds_chinese_and_english_barbecue(self):
        for post in ('今天和朋友去公园烧烤，差点把肉烤糊。', 'barbecue grill food with my friends'):
            result = find_references(post)
            self.assertEqual(result['matches'][0]['id'], 'barbecue')
            self.assertIn('AI', result['matches'][0]['source'])

    def test_unrelated_query_does_not_fabricate_references(self):
        self.assertEqual(find_references('量子纠缠与拓扑绝缘体')['matches'], [])

    def test_study_and_pet_queries_are_content_specific(self):
        self.assertEqual(find_references('论文终于提交了')['matches'][0]['id'], 'thesis')
        self.assertEqual(find_references('cat kitten sleeping on sofa')['matches'][0]['id'], 'cat')

    def test_http_writing_routes(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), WorkspaceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        try:
            connection.request('POST', '/api/references', json.dumps({'post': 'barbecue grill'}), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())['matches'][0]['id'], 'barbecue')
            with patch('server.rewrite_post', return_value={'versions': ['one', 'two', 'three']}) as rewrite:
                connection.request('POST', '/api/rewrite', json.dumps({'post': 'a post', 'style': 'humor'}), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(len(json.loads(response.read())['versions']), 3)
                rewrite.assert_called_once_with('a post', 'humor')
                connection.request('POST', '/api/rewrite', json.dumps({'post': 'a post', 'style': []}), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
                self.assertEqual(rewrite.call_count, 1)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
