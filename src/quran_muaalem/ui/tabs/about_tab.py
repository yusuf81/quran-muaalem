"""About & Credit tab for Gradio interface."""

import gradio as gr


def create_about_tab():
    """Create the About & Credit tab.
    
    Returns:
        Gradio Tab component
    """
    with gr.Tab("Tentang & Credit"):
        gr.Markdown("""
        # Tentang Aplikasi - About This Application
        
        Aplikasi ini menggunakan model **Muaalem** untuk deteksi kesalahan dan koreksi pengucapan dalam bacaan Al-Quran berdasarkan kaidah tajwid.
        
        *This application uses the **Muaalem** model for pronunciation error detection and correction in Quranic recitation based on tajweed rules.*
        
        ---
        
        ## 📄 Paper / Makalah Penelitian
        
        **Title:** Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning
        
        **Authors:** Abdullah Abdelfattah, Mahmoud I. Khalil, Hazem Abbas
        
        **Published:** arXiv preprint arXiv:2509.00094 (2025)
        
        **Links:**
        - 📖 [Read Paper on arXiv](https://arxiv.org/abs/2509.00094)
        - 🌐 [Project Website](https://obadx.github.io/prepare-quran-dataset/)
        - 📓 [PDF Version](https://arxiv.org/pdf/2509.00094)
        - 🔗 [DOI: 10.48550/arXiv.2509.00094](https://doi.org/10.48550/arXiv.2509.00094)
        
        ---
        
        ## 💻 Source Code / Kode Sumber
        
        **GitHub Repository:** [obadx/quran-muaalem](https://github.com/obadx/quran-muaalem)
        
        ---
        
        ## 📚 Citation / Sitasi
        
        Jika Anda menggunakan aplikasi ini atau model Muaalem dalam penelitian Anda, mohon sitasi paper berikut:
        
        *If you use this application or the Muaalem model in your research, please cite the following paper:*
        
        ### BibTeX
        
        ```bibtex
        @article{abdelfattah2025automatic,
          title={Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning},
          author={Abdelfattah, Abdullah and Khalil, Mahmoud I. and Abbas, Hazem},
          journal={arXiv preprint arXiv:2509.00094},
          year={2025},
          url={https://arxiv.org/abs/2509.00094},
          doi={10.48550/arXiv.2509.00094}
        }
        ```
        
        ### APA Style
        
        ```
        Abdelfattah, A., Khalil, M. I., & Abbas, H. (2025). Automatic Pronunciation Error 
        Detection and Correction of the Holy Quran's Learners Using Deep Learning. 
        arXiv preprint arXiv:2509.00094. https://doi.org/10.48550/arXiv.2509.00094
        ```

        
        ---
        
        ## 🔧 Modifications in This Version / Modifikasi dalam Versi Ini
        
        Aplikasi ini merupakan modifikasi dari repository asli dengan penambahan fitur-fitur berikut:
        
        *This application is a modified version of the original repository with the following additional features:*
        
        ### 1. 🌏 Indonesian Translation / Translasi Bahasa Indonesia
        - Interface bilingual (Indonesia & English) untuk kemudahan pengguna lokal
        - Bilingual interface (Indonesian & English) for local users' convenience
        
        ### 2. 📖 Automatic Verse Segmentation / Pemenggalan Ayat Otomatis
        - **Fitur analisis banyak ayat sekaligus** menggunakan model segmentation (`recitation-segmenter-v2`)
        - Otomatis memisahkan rekaman panjang menjadi ayat-per-ayat berdasarkan jeda (waqf)
        - Mendukung analisis full surah
        - **Multi-verse analysis feature** using segmentation model (`recitation-segmenter-v2`)
        - Automatically segments long recordings into individual verses based on pauses (waqf)
        - Supports full surah analysis
        
        ### 3. 🔇 Audio Preprocessing / Pengolahan Audio
        - Setiap segmen ayat dipotong dengan **padding 0.5 detik** di awal dan akhir untuk menjaga konteks
        - **Silence trimming** opsional dilakukan SETELAH pemenggalan untuk membersihkan keheningan
        - Each verse segment is sliced with **0.5-second padding** at the beginning and end to preserve context
        - Optional **silence trimming** is applied AFTER segmentation to remove silence
        
        ### 4. 📊 Enhanced UI / Peningkatan Antarmuka
        - Preview teks Al-Quran sebelum analisis
        - Hasil analisis per-ayat dengan timestamp
        - Quran text preview before analysis
        - Per-verse analysis results with timestamps
        
        **Modified by:** Yusuf (GitHub: [yusuf81](https://github.com/yusuf81))
        
        ---
        
        ## 🙏 Acknowledgments / Penghargaan
        
        Terima kasih kepada tim peneliti yang telah mengembangkan model Muaalem dan menyediakan dataset serta kode sumber secara open-source.
        
        *Thanks to the research team who developed the Muaalem model and provided the dataset and source code as open-source.*
        
        ---
        
        ## ⚖️ License / Lisensi
        
        Proyek ini menggunakan lisensi yang sama dengan repository aslinya. Silakan kunjungi [GitHub repository](https://github.com/obadx/quran-muaalem) untuk informasi lisensi lengkap.
        
        *This project uses the same license as the original repository. Please visit the [GitHub repository](https://github.com/obadx/quran-muaalem) for complete license information.*
        """)
