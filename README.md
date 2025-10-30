# CS 362 Assignment 4
Belief Networks & Hidden Markov Models

## What's in here

This assignment has two parts:
- Part 1: Bayesian networks using pgmpy (alarm example and car problem)
- Part 2: HMM spelling corrector with Viterbi algorithm

## Running the code

Install dependencies first:
```
pip install -r requirements.txt
```

Part 1 (Bayesian networks):
```
python3 src/alarm.py
python3 src/carnet.py
python3 src/carnet_with_key.py
```

Part 2 (spell corrector):
```
python3 src/spell_corrector.py
```

## Reflection Questions

### Question 1: Correctly spelled word that gets "corrected" wrong

Example: "quiz" → "quin", "hello" → "helly"

This is actually pretty interesting (and annoying). The word "quiz" is spelled perfectly fine but my HMM insists it should be "quin". 

What's happening is the Viterbi algorithm is calculating the maximum likelihood path through all possible letter sequences. At each position it's multiplying transition probabilities with emission probabilities. The issue is aspell.txt just doesn't have many words with 'z' in them. Especially not with 'ui' followed by 'z'.

When the algorithm gets to that third position (already processed 'ui'), it has to pick between different paths. Going through 'z' has decent emission probability since z→z is the most likely, but P(z|ui) - the transition probability - is really low. Meanwhile 'n' has slightly worse emission for the observed 'z', but P(n|ui) is way higher (think "ruin", "ruined", etc). When you do the math in log space, the 'n' path wins even though it's wrong.

Same deal with "hello" → "helly" - the double 'll' messes with transition patterns, then l→o at the end apparently isn't as common as l→y in whatever training data it learned from. The algorithm doesn't actually "know" these are real words, it's just following probabilities.

This shows the big limitation here - HMMs assume every word comes from the same Markov process, but really some words are just rare or have unusual patterns. You'd need something smarter, like a dictionary check or actual word-level model, to catch that "quiz" is a real word.

### Question 2: Misspelled word that gets "corrected" to the wrong thing

Examples: "helo" → "hely" (should be "hello"), "teh" → "tes" (should be "the")

This highlights how broken my implementation is for actual typos. The problem is I'm doing this really simple character-by-character alignment during training which just falls apart when letters are missing or swapped.

Take "helo" - it's missing an 'l'. The HMM assumes the observed sequence and hidden sequence are the same length, so it's trying to find a 4-letter word that generates "helo":
- Position 1: h → probably h
- Position 2: e → probably e
- Position 3: l → probably l  
- Position 4: o → ...and this is where everything breaks

At that last position, it needs to emit 'o' from state 'l'. But l→o transitions might not be super common in the training data. Words ending in 'ly' are way more common, so l→y probably has higher transition probability. Add in that the emission probabilities are trained on completely different error patterns (not on "helo"→"hello" specifically), and the algorithm decides 'y' gives better overall probability. Hence "hely".

"teh" → "tes" is even more broken because of the transposition. The algorithm goes:
- Position 1: 't' (fine)
- Position 2: 'e' coming from 't' (still fine)
- Position 3: needs to emit 'h' from state 'e'

But e→h transitions are pretty rare in English - how many words have 'eh'? Meanwhile e→s is super common (test, best, rest, west...). Even though h→h has better emission probability than s→h, the transition probability is so much worse that it picks 's' instead.

The real issue is Viterbi makes these greedy decisions at each step based on cumulative probability. Once it goes down the wrong path it can't backtrack. It has no way to look ahead and realize "hey this doesn't spell a real word."

You'd need something more sophisticated - maybe use edit distance during training (like Baum-Welch or dynamic time warping) to properly learn insertions/deletions. Or honestly just use a word-level language model that actually knows what valid words look like.

### Question 3: Misspelled word that actually gets fixed

Example: If I type "tge" (common transposition/substitution near 't' and 'h'), it would likely correct to "the"

This one actually has a chance of working because of how the probabilities line up. When someone types "tge" they probably meant "the" - maybe they hit 'g' instead of 'h' (they're close on the keyboard), or it's a weird transposition.

Here's why it could work: The HMM sees "tge" and tries to find the most likely hidden word. At each position:
- Position 1: 't' observed, most likely correct letter is 't' (emission t→t is highest)
- Position 2: 'g' observed, coming from state 't'. Now it needs to pick the next state that:
  - Has good transition from 't' (t→h is very common, t→g less so)
  - Has decent emission for 'g' (even if h→g isn't the highest)
- Position 3: 'e' observed, needs state with good e→e emission and good transition from position 2

The key is that the transition t→h is SO common (think the, this, that, there, then, these...) that even though the emission h→g might not be great, the overall path probability through 'h' at position 2 could win out. Then h→e is also really common, and e→e has high emission probability.

Compare this to the failures:
- Question 1: "quiz" was already correct but had rare transitions (ui→z), so it got "corrected" wrong
- Question 2: "helo" had wrong length so the alignment broke, "teh" had e→s winning over e→h

But "tge"→"the" works because:
1. Same length (3 letters) - no insertion/deletion issues
2. The correct word "the" has extremely high transition probabilities throughout
3. Even though some emission probabilities are off (h→g), they're not SO bad that they kill the path
4. No competing high-probability 3-letter sequence starting with 't' and ending with 'e'

The algorithm essentially "forgives" the middle character's emission error because the transition probabilities are so strong. It's like the model says "I know 'h' doesn't usually emit 'g', but t→h→e is such a common sequence that this is probably what they meant."

This is different from "quiz" where the transition probabilities themselves were weak, and different from "helo" where the length mismatch broke everything.

For this to work, you need:
1. The typo has the same length as the correct word
2. The correct word has very strong transition probabilities
3. The emission errors aren't too extreme
4. There's no other high-probability word that could generate the same observed sequence

That's a pretty narrow window, which is why character-level HMMs aren't great for real spell checking. They work best on simple substitution errors in very common words with strong transition patterns.

### Question 4: Real typos vs synthetic typos

This gets at something pretty fundamental - your model is only as good as your training data.

**Real typos**

If we trained on actual typos from real people (Google searches, Twitter, Reddit, whatever), we'd learn way more realistic patterns. 

Keyboard proximity errors would show up in the emission probabilities. When someone types 's' but hits 'd' instead, that's common because they're right next to each other on QWERTY. So P(d observed | s correct) would be way higher than some random letter. The aspell.txt data probably doesn't capture this - it seems more like random substitutions.

You'd also get cognitive errors - people mixing up homophones (their/there/they're) or common misspellings (recieve vs receive). These would create patterns that match how humans actually think about words, not just motor errors.

And patterns with doubled letters - people constantly mess these up (accomodate vs accommodate). The emission probabilities would reflect that people drop or add letters in doubles all the time.

Transition probabilities might even capture things like typing faster at the end of words and making more mistakes there.

But real data is messy. You'd get intentional misspellings (ur for your), slang, abbreviations, non-native speakers with totally different error patterns, British vs American spelling differences... and just complete garbage that isn't even trying to be words. You'd need heavy preprocessing and even then might overfit to your specific source (Twitter typos are different from email typos are different from phone typos).

**Synthetic typos**

Programmatic generation gives you control - you could generate every possible substitution, insertion, deletion, transposition. Perfectly balanced data for each error type. Clean emission probabilities, fast training, complete coverage.

But it misses how people actually type. A program doesn't know that people rarely transpose non-adjacent letters, or that 'q' is always followed by 'u', or that some typos feel more "natural" than others. Context matters too - you type differently when tired vs alert, phone vs keyboard.

The probabilities would be artificially uniform instead of matching real typing behavior.

**What would work better**

For something people would actually use, you'd want real data but with smart processing. Collect from multiple sources to get diverse errors. Filter heavily for quality. Maybe weight by context (professional vs casual). Then augment with synthetic data for rare cases so you don't get zero probabilities.

Emission probabilities would be way better with real data since they'd match actual typing. Transition probabilities probably wouldn't change much since letter sequences are fairly consistent regardless of spelling. But you'd get better coverage of unusual words.

The current aspell.txt seems semi-synthetic - it has real word pairs but they might be artificially created to cover common typos. That's why it handles common words ok but fails on anything weird. A production system would need millions of real examples to work properly.
