with open(f'test.out') as f:
    lines = ''.join(f.readlines()).split('PROPERTY')[1:]
    num_ver = list(map(lambda x : x.count('WARNING'), lines))
    num_truly_ver = list(filter(lambda x : x == 9, num_ver))
    print(len(num_truly_ver))