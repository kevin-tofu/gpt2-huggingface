
import os

def livedoor_categories() -> list:
    ret = [
        'dokujo-tsushin',
        'it-life-hack',
        'kaden-channel',
        'livedoor-homme',
        'movie-enter',
        'peachy',
        'smax',
        'sports-watch',
        'topic-news'
    ]
    return ret


def main(catname: str = 'livedoor-homme'):
    
    print(f'category : {catname}')
    assert catname in livedoor_categories()

    path = f"text/{catname}"
    title = []
    filenames = os.listdir(path)
    for filename in filenames:
        with open(f"{path}/{filename}") as f:
            datalist = f.readlines()
            for data in datalist:
                if data == '\n':
                    pass
                else:
                    title.append(data)

    return title


def split_title_train_val(
    title: list[str],
    ratio_train: float = 0.8,
    shuffle: bool = False,
    path_export: str = './dataset',
    name_dataset: str = 'test1'
):
    
    import os
    import random
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(path_export)
    if not isExist:
       os.makedirs(path_export)


    if shuffle is True:
        title = random.shuffle(title)

    title = [s.replace('\n', '') for s in title]
    num = len(title)
    idx_train = int(num*ratio_train)
    with open(
        f'{path_export}/{name_dataset}_train.txt',
        mode='w',
        newline=''
    ) as f:
        f.writelines(title[:idx_train])

    with open(
        f'{path_export}/{name_dataset}_val.txt',
        mode='w',
        newline=''
    ) as f:
        f.writelines(title[idx_train:])


if __name__ == '__main__':
    
    categories = livedoor_categories()
    title = main(categories[1])
    for t in title[0:20]:
        print(t)

    split_title_train_val(
        title,
        ratio_train = 0.8,
        shuffle = False,
        path_export = './dataset',
        name_dataset = 'dokujo'
    )