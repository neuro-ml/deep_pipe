def check_len(*arrays):
    length = len(arrays[0])
    for i, a in enumerate(arrays):
        assert length == len(a), f'Different len: {arrays}'
