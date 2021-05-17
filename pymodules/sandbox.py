from typing import Any, Counter, List
import unittest

class ShortRoutines:
    def is_anagram(a: str, b: str) -> bool:
        if len(a) != len(b):
            return False
        counter: List[int] = [0] * 26

        for (a_c, b_c) in zip(a, b):
            counter[ord(a_c) - ord('a')] += 1
            counter[ord(b_c) - ord('a')] -= 1
        return  all([v == 0 for v in counter])
        


class HeapEntry:
    '''
    '''

class HeapBasedMethods:
    """
     A collection of routines that rely on the heap data structure
    """

    # Given a non-empty array of integers, return the k most frequent elements.
    # 
    # You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
    # Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
    # It's guaranteed that the answer is unique, in other words the set of the top k frequent elements is unique.
    # You can return the answer in any order.
    def most_frequent_k(items: List[int], k: int) -> List[int]:
        counts = Counter(items)
        return [tuple_[0] for tuple_ in  counts.most_common(k)]

    #  Given a non-empty list of words, return the k most frequent elements.
    # 
    # Your answer should be sorted by frequency from highest to lowest.
    # If two words have the same frequency,
    # then the word with the lower alphabetical order comes first.
    def most_frequent_k_sorted(items: List[int], k: int) -> List[int]:
        pass


class TestHeapBasedMethods(unittest.TestCase):
    def test_is_anagram(self):
        self.assertEqual(True, ShortRoutines.is_anagram('carbonate', 'boncarate'))
        self.assertEqual(False, ShortRoutines.is_anagram('calcium', 'boncarate'))

class CombinatorialSearch:    
    def search(input: List[Any]) -> List[Any]:
        '''
        Given an input collection, return all combinations of the inputs that
        satisfy some given criterion
        '''
        stack: List[List[Any]] = []
        # Each value could conceivably be the start of a valid combination
        for possible_start in input:
            stack.append([possible_start])

        idx_in_input: int = 0
        while len(stack) > 0 and idx_in_input < len(input):
            # try extending each partial solution using input[idx_in_input]
            idx_in_input += 1

class DiscreteOptimization:
    def max_score_card_game(self, cards: List[int]) -> int:
        return self.card_game_helper(cards, 0)

    
    def card_game_helper(self, remaining_cards: List[int], turn_id: int) -> int:
        """
            remaining_cards is a slice of the cards.
            turn_id tells us whose turn it is. If it's even, we know its our turn
        """
        # The base case happens when we exhaust all the cards
        if len(remaining_cards) == 0:
            return 0

        # At each turn, both the opponent and we have three possible actions.
        # TakeOne, TakeTwo or TakeThree. We do not know a priori which
        # action will lead to the highest eventual score. So we should evaluate
        # all three of them and select the one that leads us to the highest score.
        #
        # to evaluate an action, we perform that action and evaluate all the resulting
        # states -- we do this recursively. 
        #
        # of course an action can only be taken if it is feasible from the current
        # state. in this instance, an action is feasible if there are enough cards
        # to perform it
        scores = [self.card_game_helper(remaining_cards[1:], turn_id + 1),
                self.card_game_helper(remaining_cards[2:], turn_id + 1),
                self.card_game_helper(remaining_cards[3:], turn_id + 1)]
        max_eventual = max(scores)
        if turn_id & 1 == 0:
            argmax_eventual = sum(remaining_cards[:scores.index(max_eventual) + 1])
            max_eventual += argmax_eventual
        return max_eventual


if __name__ == '__main__':
    # unittest.main()
    print(DiscreteOptimization().max_score_card_game([5, 7, -5, 6, -8, 9, 5]))