import os
import xml.etree.ElementTree as ET
import glob
import io
import codecs

from torchtext import data

class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
       
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
            io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            
            src_lines = src_file.read().split('\n')[:15000] # 추후 인덱싱슬라이스 제거하기
            trg_lines = trg_file.read().split('\n')[:15000]

            c = list(zip(src_lines, trg_lines))
            examples = [data.Example.fromlist([x, y], fields) for x, y in c]

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

class WMT14(TranslationDataset):
    """The WMT 2014 English-German dataset, as preprocessed by Google Brain.

    Though this download contains test sets from 2015 and 2016, the train set
    differs slightly from WMT 2015 and 2016 and significantly from WMT 2017."""

    urls = [('https://docs.google.com/uc?export=download&id=1zqVTc0sm5J-3_8OVaRrAM2X36CrRMqiS&confirm=t', 'data.zip')] # 여기 고치기
    name = 'wmt14'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
            train='train',
            validation='newstest2013',
            test='newstest2014', **kwargs):
        """Create dataset objects for splits of the WMT 2014 dataset.

        Arguments:
            exts: A tuple containing the extensions for each language. Must be
                either ('.en', '.de') or the reverse.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default:
                'train.tok.clean.bpe.32000'.
            validation: The prefix of the validation data. Default:
                'newstest2013.tok.bpe.32000'.
            test: The prefix of the test data. Default:
                'newstest2014.tok.bpe.32000'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(WMT14, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)
