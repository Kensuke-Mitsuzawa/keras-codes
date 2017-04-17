# これは何?

kerasを使って、色々と遊んでみるリポジトリです。

# Requirement

- python3 (anaconda3-4.0.0 python3.5で動作確認)

## Dependency

- Keras (2.0.3)
- tensorflow (1.0.1)
- scipy (0.19.0)
- scikit-learn (0.18.1)
- numpy (1.12.1)

# setup

anaconda3ディストリビューションの利用を推奨します。

```
python setup.py
```

## データ準備

サンプルデータの用意をします。

wikipedia記事テキストをダウンロードするために次のスクリプトを実行します。

```
$ cd examples
$ python get_wikipedia_text.py
```

## word2vecモデルのダウンロード

訓練済みword2vecモデルのダウンロードをします。

```
$ cd examples
$ sh get_wikipedia_entity_vector.sh
```

