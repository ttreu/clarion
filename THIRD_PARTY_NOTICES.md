# Third-party notices

clarion's own code is under the [MIT License](LICENSE). It uses the components below, which keep their
own licenses. clarion's source code does not contain copies of them: they are installed by the package
manager (`uv sync`) or, for `deep-filter`, downloaded on first use. A built `clarion.app` (PyInstaller)
does contain copies of the Python packages and their native libraries.

## Runtime components

| Component | Used for | License | Source |
|---|---|---|---|
| DeepFilterNet (`deep-filter` binary, DeepFilterNet3 model) | AI speech enhancement | MIT or Apache-2.0 (dual) | https://github.com/Rikorose/DeepFilterNet |
| noisereduce | spectral noise reduction (music) | MIT | https://github.com/timsainb/noisereduce |
| nara_wpe | dereverberation (WPE) | MIT | https://github.com/fgnt/nara_wpe |
| PySide6 / Shiboken6 (Qt for Python) and Qt 6 | user interface, audio playback | LGPL-3.0 (also GPL-2.0 / GPL-3.0) | https://pyside.org · https://code.qt.io |
| FFmpeg (inside Qt Multimedia) | audio playback | LGPL-2.1 or later | https://ffmpeg.org |
| NumPy | numerics | BSD-3-Clause (and bundled parts under 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | filters, signal processing | BSD-3-Clause | https://scipy.org |
| python-soundfile | reading / writing audio files | BSD-3-Clause | https://github.com/bastibe/python-soundfile |
| libsndfile (bundled in python-soundfile wheels) | audio file formats | LGPL-2.1 or later | https://github.com/libsndfile/libsndfile |

Packages pulled in by the ones above (e.g. matplotlib, tqdm, joblib, cffi) are under permissive licenses
(BSD, MIT, PSF, MPL-2.0); `uv.lock` lists the exact versions.

Optional, not shipped: [ffmpeg](https://ffmpeg.org) on your system is used to decode formats such as
M4A if it is installed.

## Notes on the LGPL components (only relevant when distributing a built app)

Qt/PySide6, FFmpeg and libsndfile are licensed under the GNU Lesser General Public License. clarion uses
them unmodified, as separate dynamic libraries. Anyone distributing a built `clarion.app` should:

- include this file and the LGPL license texts (https://www.gnu.org/licenses/lgpl-3.0.html,
  https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html);
- point to the source code of the exact versions used (the links above; versions in `uv.lock`);
- keep the libraries replaceable (PyInstaller ships them as separate files inside the app bundle).

## Scientific references

- H. Schröter, T. Rosenkranz, A. N. Escalante-B., A. Maier: *DeepFilterNet: Perceptually Motivated
  Real-Time Speech Enhancement*. INTERSPEECH 2023.
- H. Schröter, A. N. Escalante-B., T. Rosenkranz, A. Maier: *DeepFilterNet: A Low Complexity Speech
  Enhancement Framework for Full-Band Audio based on Deep Filtering*. ICASSP 2022.
- T. Sainburg, M. Thielk, T. Q. Gentner: *Finding, visualizing, and quantifying latent structure across
  diverse animal vocal repertoires*. PLoS Computational Biology 16(10): e1008228, 2020.
  Software: T. Sainburg, *noisereduce*, Zenodo, https://doi.org/10.5281/zenodo.3243139
- L. Drude, J. Heymann, C. Boeddeker, R. Haeb-Umbach: *NARA-WPE: A Python package for weighted
  prediction error dereverberation in Numpy and Tensorflow for online and offline processing*.
  13. ITG Fachtagung Sprachkommunikation, 2018.
- T. Nakatani, T. Yoshioka, K. Kinoshita, M. Miyoshi, B.-H. Juang: *Speech Dereverberation Based on
  Variance-Normalized Delayed Linear Prediction*. IEEE Transactions on Audio, Speech, and Language
  Processing 18(7), 2010 (the WPE method).
- ITU-R BS.1770: *Algorithms to measure audio programme loudness and true-peak audio level*.
- EBU R 128: *Loudness normalisation and permitted maximum level of audio signals*.
- S. H. Linkwitz: *Active Crossover Networks for Noncoincident Drivers*. JAES 24(1), 1976 (crossover
  filters of the multiband compressor).

## Trademarks

Adobe and Audition are trademarks of Adobe Inc. The "broadcast" compressor uses the parameter values of
the "Broadcast" preset of Adobe Audition's Multiband Compressor as shown in that application; no Adobe
code or files are used. clarion is not affiliated with or endorsed by Adobe or by the authors of the
components above.
