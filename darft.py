MAX_WORDS_PER_TRAINING = 5

# Divide list into chunks of size n
def divide_chunks(l, n):
    return [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n )]


# Divide list into equal chunks (by adding Mockup word) 
def equal_chunks(l, n):
    chunks = divide_chunks(l, n)
    for chunk in chunks:
        if len(chunk) != MAX_WORDS_PER_TRAINING:
            while (len(chunk) < MAX_WORDS_PER_TRAINING):
                chunk.append('<unk>')

    return chunks

words = []
list_lines = ['i didnt feel humiliated', 'i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake', 'im grabbing a minute to post i feel greedy wrong', 'i am ever feeling nostalgic about the fireplace i will know that it is still on the property', 'i am feeling grouchy', 'ive been feeling a little burdened lately wasnt sure why that was']

for line in list_lines:
    words.append(line.split())
    
    
print(words)

tensors = equal_chunks(words, MAX_WORDS_PER_TRAINING)
print(tensors)