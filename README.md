# PreCOMET

The PreCOMET is a fork of [Unbabel/COMET](https://github.com/Unbabel/COMET/) from version 2.2.4.
All original licensing applies.

It is used in the [subset2evaluate](https://github.com/zouharvi/subset2evaluate) package.
The divergent fork was created such that this package does not conflict with the original one.


## Models

The models are available on Huggingface:

- [PreCOMET-cons](https://huggingface.co/zouharvi/PreCOMET-cons)
- [PreCOMET-diversity](https://huggingface.co/zouharvi/PreCOMET-diversity)
- [PreCOMET-avg](https://huggingface.co/zouharvi/PreCOMET-avg)
- [PreCOMET-var](https://huggingface.co/zouharvi/PreCOMET-var)
- [PreCOMET-diffdisc_direct](https://huggingface.co/zouharvi/PreCOMET-diffdisc_direct)
- [PreCOMET-diff](https://huggingface.co/zouharvi/PreCOMET-diff)
- [PreCOMET-disc](https://huggingface.co/zouharvi/PreCOMET-disc)

The model usage is described on Huggingface.
Briefly, install the PreCOMET package:
```bash
pip install pip3 install git+https://github.com/zouharvi/PreCOMET.git
```

then:
```python
import precomet
model = precomet.load_from_checkpoint(precomet.download_model("zouharvi/PreCOMET-diversity"))
model.predict([
  {"src": "This is an easy source sentence."},
  {"src": "this is a much more complicated source sen-tence that will pro·bably lead to loww scores 🤪"}
])["scores"]
> [25.921934127807617, 20.805429458618164]
```

For PreCOMET-diversity, segments with lower scores are better for evaluation because they lead to different system translations.

## Other

This work is described in [How to Select Datapoints for Efficient Human Evaluation of NLG Models?](https://arxiv.org/abs/2501.18251).
Cite as:
```
@misc{zouhar2025selectdatapointsefficienthuman,
    title={How to Select Datapoints for Efficient Human Evaluation of NLG Models?}, 
    author={Vilém Zouhar and Peng Cui and Mrinmaya Sachan},
    year={2025},
    eprint={2501.18251},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2501.18251}, 
}
```