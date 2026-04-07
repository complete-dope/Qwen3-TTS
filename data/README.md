---
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: text
    dtype: string
  - name: gender
    dtype:
      class_label:
        names:
          '0': female
          '1': male
  splits:
  - name: train
    num_bytes: 9067255387
    num_examples: 11825
  download_size: 8210562900
  dataset_size: 9067255387
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
language:
- hi
pretty_name: Hindi Indic TTS Dataset
size_categories:
- 10K<n<100K
task_categories:
- text-to-speech
---

# Hindi Indic TTS Dataset

This dataset is derived from the Indic TTS Database project, specifically using the Hindi monolingual recordings from both male and female speakers. The dataset contains high-quality speech recordings with corresponding text transcriptions, making it suitable for text-to-speech (TTS) research and development.

## Dataset Details

- **Language**: Hindi
- **Total Duration**: ~10.33 hours (Male: 5.16 hours, Female: 5.18 hours)
- **Audio Format**: WAV
- **Sampling Rate**: 48000Hz
- **Speakers**: 2 (1 male, 1 female native Hindi speakers)
- **Content Type**: Monolingual Hindi utterances
- **Recording Quality**: Studio-quality recordings
- **Transcription**: Available for all audio files

## Dataset Source

This dataset is derived from the Indic TTS Database, a special corpus of Indian languages developed by the Speech Technology Consortium at IIT Madras. The original database covers 13 major languages of India and contains 10,000+ spoken sentences/utterances for both monolingual and English recordings.

## License & Usage

This dataset is subject to the original Indic TTS license terms. Before using this dataset, please ensure you have read and agreed to the [License For Use of Indic TTS](https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf).

## Acknowledgments

This dataset would not be possible without the work of the Speech Technology Consortium at IIT Madras. Special acknowledgment goes to:
- Speech Technology Consortium
- Department of Computer Science & Engineering and Electrical Engineering, IIT Madras
- Bhashini, MeitY
- Prof. Hema A Murthy & Prof. S Umesh

## Citation

If you use this dataset in your research or applications, please cite the original Indic TTS project:

```bibtex
@misc{indictts2023,
    title = {Indic {TTS}: A Text-to-Speech Database for Indian Languages},
    author = {Speech Technology Consortium and {Hema A Murthy} and {S Umesh}},
    year = {2023},
    publisher = {Indian Institute of Technology Madras},
    url = {https://www.iitm.ac.in/donlab/indictts/},
    institution = {Department of Computer Science and Engineering and Electrical Engineering, IIT MADRAS}
}
```

## Contact

For any issues or queries related to this HuggingFace dataset version, feel free to comment in the Community tab.

For queries related to the original Indic TTS database, please contact: smtiitm@gmail.com

## Original Database Access

The original complete database can be accessed at: https://www.iitm.ac.in/donlab/indictts/database

Note: The original database provides access to data in multiple Indian languages and variants. This HuggingFace dataset specifically contains the Hindi monolingual portion of that database.