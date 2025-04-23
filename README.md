# Keep It PG or Let It Go?  
**Exploring the Presence of (In)appropriateness in YouTube Videos for Young Children and Opportunities for Safeguarding**

This repository contains the code and supporting materials for my master's thesis on detecting and addressing inappropriate content in YouTube videos recommended to young children. The project explores how metadata-derived features can support classification and reranking strategies to improve content safety on video platforms like YouTube and YouTube Kids.

---

## Thesis

**Title:** Keep It PG or Let It Go? Exploring the Presence of (In)appropriateness in YouTube Videos for Young Children and Opportunities for Safeguarding  
**Author:** J.J.P. de Water
**Institution:** TU Delft  
**Date:** 2024  
**Repository link:** [TU Delft Repository](https://repository.tudelft.nl/record/uuid:51cb02cc-0483-43f9-9d11-4289e5de3fd7)

---

## Project Overview

The study investigates whether a nuanced understanding of video (in)appropriateness, categorized as suitable, irrelevant, restricted, or disturbing, can improve content filtering and recommendation outcomes for young children aged 0-5. It includes:

- A feature analysis using video metadata (e.g., tags, description sentiment, engagement)
- A classification model to predict child-appropriateness types
- Score-based reranking strategies to improve YouTube recommendations

The goal is to reduce the exposure of young children to inappropriate content while promoting appropriate material.

---

## Usage

You can install dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

---

## Citation

If you use or refer to this codebase, please cite the thesis as:

```
@mastersthesis{dewater2024thesis,
  title={Keep It PG or Let It Go? Exploring the Presence of (In)appropriateness in YouTube Videos for Young Children and Opportunities for Safeguarding},
  author={J. de Water},
  school={TU Delft},
  year={2024},
  url={https://repository.tudelft.nl/}
}
```

This project makes use of a **phonemic decoding model created by Pinney et al. (2024)** for feature extraction. Please cite their work as:

```
@inproceedings{pinney2024incorporating,
  title={Incorporating Word-level Phonemic Decoding into Readability Assessment},
  author={Pinney, Christine and Kennington, Casey and Pera, Maria Soledad and Wright, Katherine Landau and Fails, Jerry Alan},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={8998--9009},
  year={2024}
}
```

and

```
@inproceedings{pinney2024readability,
  title={How Readability Cues Affect Children's Navigation of Search Engine Result Pages},
  author={Pinney, Christine and Bettencourt, Benjamin J and Fails, Jerry Alan and Kennington, Casey and Wright, Katherine Landau and Pera, Maria Soledad},
  booktitle={Proceedings of the 23rd Annual ACM Interaction Design and Children Conference},
  pages={62--69},
  year={2024}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
