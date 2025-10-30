"""
hmm spelling corrector using viterbi algorithm.
"""

import math
from collections import defaultdict

class HMMSpellCorrector:
    def __init__(self, training_file):
        self.emission_counts = defaultdict(lambda: defaultdict(int))  # emission_counts[correct_letter][typed_letter]
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.transition_counts = defaultdict(lambda: defaultdict(int))  # transition_counts[from_letter][to_letter]
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.start_counts = defaultdict(int)  # start_counts[letter]
        self.start_probs = defaultdict(float)
        self.end_counts = defaultdict(int)  # end_counts[letter]
        self.end_probs = defaultdict(float)
        
        self.letters = set()
        
        self._load_training_data(training_file)
        self._calculate_emission_probabilities()
        self._calculate_transition_probabilities()
    
    def _load_training_data(self, training_file):
        with open(training_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(':')
                if len(parts) != 2:
                    continue
                
                correct_word = parts[0].strip().lower()
                typo_words = parts[1].strip().split()
                
                self._process_correct_word(correct_word)
                
                for typo_word in typo_words:
                    typo_word = typo_word.lower()
                    self._process_typo_pair(correct_word, typo_word)
    
    def _process_correct_word(self, word):
        if not word:
            return
        
        for letter in word:
            self.letters.add(letter)
        
        self.start_counts[word[0]] += 1
        
        for i in range(len(word) - 1):
            self.transition_counts[word[i]][word[i+1]] += 1
        
        self.end_counts[word[-1]] += 1
    
    def _process_typo_pair(self, correct_word, typo_word):
        # simple character-by-character alignment
        min_len = min(len(correct_word), len(typo_word))
        
        for i in range(min_len):
            correct_char = correct_word[i]
            typed_char = typo_word[i]
            self.emission_counts[correct_char][typed_char] += 1
    
    def _calculate_emission_probabilities(self):
        # calculate emission probabilities with smoothing
        all_typed_letters = set()
        for correct_letter in self.emission_counts:
            all_typed_letters.update(self.emission_counts[correct_letter].keys())
        all_typed_letters.update(self.letters)
        
        smoothing_factor = 0.001
        vocab_size = len(all_typed_letters)
        
        for correct_letter in self.letters:
            total_count = sum(self.emission_counts[correct_letter].values())
            
            if total_count == 0:
                for typed_letter in all_typed_letters:
                    self.emission_probs[correct_letter][typed_letter] = 1.0 / vocab_size
            else:
                total_with_smoothing = total_count + smoothing_factor * vocab_size
                
                for typed_letter in all_typed_letters:
                    count = self.emission_counts[correct_letter].get(typed_letter, 0)
                    self.emission_probs[correct_letter][typed_letter] = \
                        (count + smoothing_factor) / total_with_smoothing
    
    def _calculate_transition_probabilities(self):
        # calculate transition probabilities with smoothing
        smoothing_factor = 0.001
        vocab_size = len(self.letters)
        
        total_start = sum(self.start_counts.values())
        total_start_with_smoothing = total_start + smoothing_factor * vocab_size
        
        for letter in self.letters:
            count = self.start_counts.get(letter, 0)
            self.start_probs[letter] = (count + smoothing_factor) / total_start_with_smoothing
        
        for from_letter in self.letters:
            total_count = sum(self.transition_counts[from_letter].values())
            total_with_smoothing = total_count + smoothing_factor * vocab_size
            
            for to_letter in self.letters:
                count = self.transition_counts[from_letter].get(to_letter, 0)
                self.transition_probs[from_letter][to_letter] = \
                    (count + smoothing_factor) / total_with_smoothing
        
        total_end = sum(self.end_counts.values())
        total_end_with_smoothing = total_end + smoothing_factor * vocab_size
        
        for letter in self.letters:
            count = self.end_counts.get(letter, 0)
            self.end_probs[letter] = (count + smoothing_factor) / total_end_with_smoothing
    
    def viterbi(self, observed_word):
        # viterbi algorithm for spelling correction
        observed_word = observed_word.lower()
        
        if not observed_word:
            return ""
        
        if len(observed_word) == 1:
            best_letter = observed_word[0]
            best_prob = float('-inf')
            
            for correct_letter in self.letters:
                prob = math.log(self.start_probs.get(correct_letter, 1e-10)) + \
                       math.log(self.emission_probs[correct_letter].get(observed_word[0], 1e-10))
                
                if prob > best_prob:
                    best_prob = prob
                    best_letter = correct_letter
            
            return best_letter
        
        viterbi = [defaultdict(lambda: float('-inf')) for _ in range(len(observed_word))]
        backpointer = [defaultdict(str) for _ in range(len(observed_word))]
        
        for state in self.letters:
            emission_prob = self.emission_probs[state].get(observed_word[0], 1e-10)
            start_prob = self.start_probs.get(state, 1e-10)
            viterbi[0][state] = math.log(start_prob) + math.log(emission_prob)
        
        for t in range(1, len(observed_word)):
            observed_letter = observed_word[t]
            
            for current_state in self.letters:
                emission_prob = self.emission_probs[current_state].get(observed_letter, 1e-10)
                
                best_prob = float('-inf')
                best_prev_state = None
                
                for prev_state in self.letters:
                    transition_prob = self.transition_probs[prev_state].get(current_state, 1e-10)
                    prob = viterbi[t-1][prev_state] + math.log(transition_prob) + math.log(emission_prob)
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_prev_state = prev_state
                
                viterbi[t][current_state] = best_prob
                backpointer[t][current_state] = best_prev_state
        
        best_final_state = None
        best_final_prob = float('-inf')
        
        for state in self.letters:
            prob = viterbi[len(observed_word)-1][state] + math.log(self.end_probs.get(state, 1e-10))
            
            if prob > best_final_prob:
                best_final_prob = prob
                best_final_state = state
        
        path = [best_final_state]
        current_state = best_final_state
        
        for t in range(len(observed_word) - 1, 0, -1):
            prev_state = backpointer[t][current_state]
            path.insert(0, prev_state)
            current_state = prev_state
        
        return ''.join(path)
    
    def correct_text(self, text):
        words = text.split()
        corrected_words = []
        
        for word in words:
            corrected = self.viterbi(word)
            corrected_words.append(corrected)
        
        return ' '.join(corrected_words)
    
    def correct_interactive(self):
        print("=" * 60)
        print("HMM SPELLING CORRECTOR")
        print("=" * 60)
        print("Enter text to correct (or 'quit' to exit)")
        print()
        
        while True:
            user_input = input("Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            corrected = self.correct_text(user_input)
            print(f"Corrected: {corrected}")
            print()


def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aspell_path = os.path.join(script_dir, '..', 'data', 'aspell.txt')
    corrector = HMMSpellCorrector(aspell_path)
    
    corrector.correct_interactive()


if __name__ == "__main__":
    main()

