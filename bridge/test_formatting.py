import unittest
from formatting import code_spans


class Formatting(unittest.TestCase):
    def test_code_entities_use_utf16_offsets_and_leave_links_intact(self):
        body = '⚠️ 😀 使用者：5172206551\n案件：#45fbe1c0\nsync:63f64489a73d\n/sudo list\nhttps://t.me/c/4453784300/27'
        raw = body.encode('utf-16-le')
        tokens = [raw[start*2:(start+length)*2].decode('utf-16-le') for start,length in code_spans(body)]
        self.assertEqual(tokens, ['5172206551', '#45fbe1c0', 'sync:63f64489a73d', '/sudo'])
        self.assertEqual(code_spans('<b>name</b> _reason_ *text*'), [])

    def test_full_case_uuid_is_one_entity(self):
        case = '5492f9d3-354a-4ff5-9131-619ded88d056'
        self.assertEqual(code_spans(case), [(0, len(case))])
