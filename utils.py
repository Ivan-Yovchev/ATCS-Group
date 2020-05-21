import random

def persuasiveness_scheduler(epoch, epochs):

    t = epoch/epochs

    # equally split time into sampling distances of 5, 4, 3, 2, and 1
    if t<=0.2:
        return [0, 5]
    elif t<=0.4:
        return random.sample(
            [
                [0,4],
                [1,5]
            ],
            1
        )[0]
    elif t<=0.6:
        return random.sample(
            [
                [0,3],
                [1,4],
                [2,5]
            ],
            1
        )[0]
    elif t<=0.8:
        return random.sample(
            [
                [0,2],
                [1,3],
                [2,4],
                [3,5]
            ],
            1
        )[0]
    else:
        return random.sample(
            [
                [0,1],
                [1,2],
                [2,3],
                [3,4],
                [4,5]
            ],
            1
        )[0]
