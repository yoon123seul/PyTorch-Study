import spacy
from torchtext.data import Field, BucketIterator
from dataloader import Multi30k, WMT14

def preprocess():
    for i, token in enumerate(tokenized):
        print(f"인덱스 {i}: {token.text}")

    def tokenize_space(text):
        return text.split(' ')

    SRC = Field(tokenize=tokenize_space, init_token="", eos_token="", lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_space, init_token="", eos_token="", lower=True, batch_first=True)

    train_dataset, valid_dataset, test_dataset = WMT14.splits(exts=(".de", ".en"), fields=(SRC, TRG))

    print(f"학습 데이터셋(training dataset) 크기: {len(train_dataset.examples)}개")
    print(f"평가 데이터셋(validation dataset) 크기: {len(valid_dataset.examples)}개")
    print(f"테스트 데이터셋(testing dataset) 크기: {len(test_dataset.examples)}개")

    # 학습 데이터 중 하나를 선택해 출력
    print(vars(train_dataset.examples[30])['src'])
    print(vars(train_dataset.examples[30])['trg'])

    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    print(f"len(SRC): {len(SRC.vocab)}")
    print(f"len(TRG): {len(TRG.vocab)}")

    print(TRG.vocab.stoi["abcabc"]) # 없는 단어: 0
    print(TRG.vocab.stoi[TRG.pad_token]) # 패딩(padding): 1
    print(TRG.vocab.stoi[""]) # : 2
    print(TRG.vocab.stoi[""]) # : 3
    print(TRG.vocab.stoi["hello"])
    print(TRG.vocab.stoi["world"])

    return SRC, TRG, train_dataset, valid_dataset, test_dataset
