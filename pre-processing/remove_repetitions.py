import nltk

def tuple_to_str(tuple):
    output_str = ""
    for tup in tuple:
        try:
            output_str += " " + tup[0]
            output_str += " " + tup[1]
        except:
            continue
    return output_str[1:]


def remove_single_repetitions(text):
    """
    Removes duplicated words (stuttering) and duplicated pauses from utterance.
    e.g: I I I I I I wanted --> I wanted.
    :param line: input text containing dupes.
    :return: unduped string containing text.
    """
    utterance = text.split(" ")

    newlist = []
    newlist.append(utterance[0])
    for i, element in enumerate(utterance):
        if i > 0 and utterance[i - 1] != element:
            newlist.append(element)

    return ' '.join(newlist)


def remove_bigram_repetitions(text):
    """
    Removes bigram stuttering from text. I went I went to the to the doctor --> I went to the doctor.
    :param text: input text containing a string
    :return: string without duplicates.
    """
    bigram = list(nltk.bigrams(text.split()))
    grams = []

    for i in range(0, len(bigram)):
        if i % 2 == 0:
            grams.append(bigram[i])

    result = []
    prev_item = None
    for item in grams:
        if item != prev_item:
            result.append(item)
            prev_item = item

    if result[-1][-1] != bigram[-1][-1]:
        result.append(tuple((bigram[-1][-1]).split(" ")))

    return tuple_to_str(result)

def remove_all_repetitions(text):
    """
    Removes bigram repetitions and stuttering from text.
    :return: Clean text.
    """
    try:
        output_text = remove_single_repetitions(text)
        output_text2 = remove_bigram_repetitions(output_text)
    except:
        return text
    return output_text2
