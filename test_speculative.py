"""Algorithmic tests independent of Vulkan/model weights."""
import unittest
import numpy as np
import speculative


class ToyEngine:
    max_ctx = 40
    def __init__(self, continuation):
        self.continuation = continuation
        self.prefix = []
        self.calls = []
        self.cache_length = 0

    def clear_cache(self):
        self.prefix = []
        self.cache_length = 0

    def truncate_cache(self, length):
        assert 0 <= length <= self.cache_length
        self.prefix = self.prefix[:length]
        self.cache_length = length

    def forward(self, ids, pos_start=0, all_logits=False):
        assert pos_start == self.cache_length
        assert pos_start+len(ids) <= self.max_ctx
        self.calls.append(list(ids))
        rows=[]
        for token in ids:
            self.prefix.append(token)
            self.cache_length += 1
            probs=np.full(16,-10.,np.float32)
            probs[self.continuation(self.prefix)] = 10
            rows.append(probs)
        return np.array(rows) if all_logits else rows[-1]


class SpeculativeTests(unittest.TestCase):
    def test_longest_most_recent_ngram(self):
        self.assertEqual(speculative.ngram_draft([1,2,3,4,1,2],3,4),[3,4,1])
        self.assertEqual(speculative.ngram_draft([1,2,7,1,2,8,1,2],1,2),[8])
        self.assertEqual(speculative.ngram_draft([1,2,3],4,4),[])

    def test_full_acceptance_and_budget(self):
        e=ToyEngine(lambda p: {1:2,2:3,3:1}[p[-1]])
        seen=[];stats={}
        got=speculative.generate_greedy(e,[1,2,3]*3,max_new=11,eos=15,
                                       callback=seen.append,block_size=4,stats=stats)
        self.assertEqual(got,([1,2,3]*4)[:11])
        self.assertEqual(seen,got)
        self.assertEqual(e.cache_length,20)
        self.assertTrue(any(len(x)>1 for x in e.calls[1:]))
        self.assertEqual(stats['accepted_draft'],8)

    def test_reject_and_cache_rollback(self):
        # History suggests 1,2,3 again, but the true continuation after 1 is 9.
        e=ToyEngine(lambda p: 1 if len(p)==6 else 9)
        got=speculative.generate_greedy(e,[1,2,3,1,2,3],max_new=6,eos=15,
                                       block_size=4)
        self.assertEqual(got,[1,9,9,9,9,9])
        self.assertEqual(e.prefix,[1,2,3,1,2,3]+got)

    def test_partial_acceptance_then_rejection(self):
        e=ToyEngine(lambda p: 1 if len(p)==6 else 2 if len(p)==7 else 9)
        stats={}
        got=speculative.generate_greedy(e,[1,2,3,1,2,3],max_new=6,eos=15,
                                       block_size=4,stats=stats)
        self.assertEqual(got,[1,2,9,9,9,9])
        self.assertEqual(e.prefix,[1,2,3,1,2,3]+got)
        self.assertEqual(e.calls[1],[1,2,3,1])
        self.assertEqual(e.calls[2][0],9)

    def test_rejection_backoff(self):
        # For a guaranteed consecutive rejection, supply an intentionally bad
        # proposal while still exercising the real verification/cache loop.
        original=speculative.ngram_draft
        speculative.ngram_draft=lambda history,max_new,max_ngram: [7]*max_new
        try:
            e=ToyEngine(lambda p:9)
            stats={}
            got=speculative.generate_greedy(e,[1],max_new=12,eos=15,stats=stats)
        finally:
            speculative.ngram_draft=original
        self.assertEqual(got,[9]*12)
        self.assertEqual(stats['backoff_cycles'],8)
        self.assertTrue(all(len(x)==1 for x in e.calls[3:11]))

    def test_eos_first_and_inside_verified_draft(self):
        e=ToyEngine(lambda p:15)
        self.assertEqual(speculative.generate_greedy(e,[1],max_new=4,eos=15),[])
        self.assertEqual(len(e.calls),1)
        e=ToyEngine(lambda p:{1:2,2:3,3:1}[p[-1]])
        got=speculative.generate_greedy(e,[1,2,3]*3,max_new=8,eos=2,block_size=4)
        self.assertEqual(got,[1])
        self.assertEqual(e.cache_length,10)

    def test_no_match_and_context_limit(self):
        e=ToyEngine(lambda p:(p[-1]+1)%16)
        e.max_ctx=6
        got=speculative.generate_greedy(e,[1,2,3],max_new=20,eos=-1)
        self.assertEqual(got,[4,5,6])
        self.assertEqual(e.cache_length,6)
        self.assertEqual([len(x) for x in e.calls[1:]],[1,1,1])

    def test_zero_and_invalid_options(self):
        e=ToyEngine(lambda p:1)
        self.assertEqual(speculative.generate_greedy(e,[1],max_new=0),[])
        self.assertEqual(e.calls,[])
        for kwargs in ({'block_size':0},{'block_size':9},{'max_ngram':0},{'max_new':-1}):
            with self.assertRaises(ValueError):
                speculative.generate_greedy(e,[1],**kwargs)


if __name__=='__main__':
    unittest.main()
