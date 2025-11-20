import diff_match_patch as dmp

from .explain import expalin_sifat


def generate_sifat_explanation(table):
    """Generate Indonesian text explanations for sifat comparison results"""
    if not table:
        return ""
    
    explanations = []
    
    # Kamus penjelasan istilah sifat huruf untuk orang awam
    sifat_explanations = {
        "hams": "suara lembut tanpa dengung (seperti angin berhembus)",
        "jahr": "suara kuat dengan getaran (seperti suara gemuruh)",
        "shadeed": "bacaan kuat dan tegas (huruf harus ditekan)",
        "between": "bacaan sedang antara kuat dan lembut",
        "rikhw": "bacaan lembut dan ringan",
        "mofakham": "bacaan tebal dan berat (seperti suara dalam)",
        "moraqaq": "bacaan tipis dan ringan",
        "low_mofakham": "bacaan agak tebal (sedikit berat)",
        "monfateh": "mulut terbuka lebar saat membaca",
        "motbaq": "mulut tertutup saat membaca",
        "safeer": "ada desisan seperti siulan",
        "no_safeer": "tidak ada desisan",
        "moqalqal": "suara bergetar/bergoyang",
        "not_moqalqal": "suara stabil tanpa getaran",
        "mokarar": "ada pengulangan suara",
        "not_mokarar": "tidak ada pengulangan",
        "motafashie": "suara menyebar/meluas",
        "not_motafashie": "suara tidak menyebar",
        "mostateel": "bacaan panjang dan memanjang",
        "not_mostateel": "bacaan pendek/tidak memanjang",
        "maghnoon": "ada dengung (ghunnah)",
        "not_maghnoon": "tidak ada dengung",
        # Tambahkan penjelasan untuk sifat lainnya
        "tafkheem_or_taqeeq": "penebalan (tafkheem) atau penipisan (tarqeeq) huruf",
        "idgham_or_izhar": "peluluran (idgham) atau pengucapan jelas (izhar)",
        "sakt_or_idraj": "jeda (sakt) atau tanpa jeda (idraj)",
        "wasl_or_waqf": "bersambung (wasl) atau berhenti (waqf)",
        "ithbat_or_hadhf": "penetapan (ithbat) atau penghapusan (hadhf)"
    }
    
    for row in table:
        tag = row["tag"]
        phoneme = row["phonemes"]
        
        if tag == "exact":
            # Check for mismatches in exact rows
            mismatches = []
            for key in row.keys():
                if key.startswith("exp_") and key != "exp_phonemes":
                    base_key = key.replace("exp_", "")
                    if row.get(key) != row.get(base_key):
                        mismatches.append((base_key, row.get(base_key), row.get(key)))
            
            if mismatches:
                exp_phoneme = row.get("exp_phonemes", phoneme)
                for base_key, actual, expected in mismatches:
                    # Skip jika nilai aktual adalah PAD (ini adalah masalah model)
                    if actual == "[PAD]" or actual == "PAD":
                        explanations.append(
                            f"⚠️ <strong>Huruf '{phoneme}'</strong>: {base_key} tidak terdeteksi oleh model"
                        )
                        continue
                    
                    # Skip jika nilai expected adalah PAD (masalah data referensi)
                    if expected == "[PAD]" or expected == "PAD":
                        explanations.append(
                            f"⚠️ <strong>Huruf '{phoneme}'</strong>: {base_key} tidak tersedia dalam referensi"
                        )
                        continue
                    
                    # Terjemahkan nama sifat
                    sifat_names = {
                        "hams": "sifat hams", "jahr": "sifat jahr", 
                        "shadeed": "sifat shadeed", "between": "sifat antara",
                        "rikhw": "sifat rikhw", "mofakham": "sifat mofakham",
                        "moraqaq": "sifat moraqaq", "low_mofakham": "sifat low mofakham",
                        "monfateh": "sifat monfateh", "motbaq": "sifat motbaq",
                        "safeer": "sifat safeer", "no_safeer": "sifat no safeer",
                        "moqalqal": "sifat moqalqal", "not_moqalqal": "sifat not moqalqal",
                        "mokarar": "sifat mokarar", "not_mokarar": "sifat not mokarar",
                        "motafashie": "sifat motafashie", "not_motafashie": "sifat not motafashie",
                        "mostateel": "sifat mostateel", "not_mostateel": "sifat not mostateel",
                        "maghnoon": "sifat maghnoon", "not_maghnoon": "sifat not maghnoon",
                        # Tambahkan sifat lainnya
                        "tafkheem_or_taqeeq": "sifat tafkheem atau tarqeeq",
                        "idgham_or_izhar": "sifat idgham atau izhar", 
                        "sakt_or_idraj": "sifat sakt atau idraj",
                        "wasl_or_waqf": "sifat wasl atau waqf",
                        "ithbat_or_hadhf": "sifat ithbat atau hadhf"
                    }
                    
                    # Terjemahkan nilai sifat
                    value_translations = {
                        "hams": "همس", "jahr": "جهر", 
                        "shadeed": "شديد", "between": "بين الشدة والرخاوة",
                        "rikhw": "رخو", "mofakham": "مفخم",
                        "moraqaq": "مرقق", "low_mofakham": "أدنى المفخم",
                        "monfateh": "منفتح", "motbaq": "مطبق",
                        "safeer": "صفير", "no_safeer": "لا صفير",
                        "moqalqal": "مقلقل", "not_moqalqal": "لا قلقلة",
                        "mokarar": "مكرر", "not_mokarar": "لا تكرار",
                        "motafashie": "متفشي", "not_motafashie": "لا تفشي",
                        "mostateel": "مستطيل", "not_mostateel": "لا إستطالة",
                        "maghnoon": "مغن", "not_maghnoon": "لا غنة",
                        # Tambahkan nilai untuk sifat lainnya
                        "tafkheem": "تفخيم", "tarqeeq": "ترقيق",
                        "idgham": "إدغام", "izhar": "إظهار", 
                        "sakt": "سكت", "idraj": "إدراج",
                        "wasl": "وصل", "waqf": "وقف",
                        "ithbat": "إثبات", "hadhf": "حذف",
                        "None": "-", "[PAD]": "-", "PAD": "-"
                    }
                    
                    sifat_name = sifat_names.get(base_key, base_key)
                    actual_translated = value_translations.get(actual, actual)
                    expected_translated = value_translations.get(expected, expected)
                    
                    # Tambahkan penjelasan untuk orang awam
                    actual_explanation = sifat_explanations.get(actual, "")
                    expected_explanation = sifat_explanations.get(expected, "")
                    
                    explanation_text = f"❌ <strong>Huruf '{phoneme}'</strong>: {sifat_name} tidak sesuai. "
                    explanation_text += f"Dibaca: <span style='color: #ff0000;'>{actual_translated}</span>"
                    if actual_explanation:
                        explanation_text += f" ({actual_explanation})"
                    explanation_text += f", Seharusnya: <span style='color: #00ff00;'>{expected_translated}</span>"
                    if expected_explanation:
                        explanation_text += f" ({expected_explanation})"
                    
                    explanations.append(explanation_text)
        
        elif tag == "insert":
            explanations.append(
                f"⚠️ <strong>Huruf '{phoneme}'</strong>: tambahan yang tidak ada dalam referensi"
            )
    
    if not explanations:
        explanations.append("✅ Semua sifat huruf sesuai dengan referensi")
    
    return "<br>".join(explanations)


def explain_for_gradio(
    phonemes: str,
    exp_phonemes: str,
    sifat: list,
    exp_sifat: list,
    original_text: str = "",
) -> str:
    # Create diff-match-patch object
    dmp_obj = dmp.diff_match_patch()

    # Calculate differences using Google's diff-match-patch (same as terminal)
    diffs = dmp_obj.diff_main(exp_phonemes, phonemes)

    # Create HTML for phoneme differences
    phoneme_html = explain_phonemes_html(dmp_obj, diffs)

    # Build sifat comparison table (used for textual explanations only)
    sifat_table = expalin_sifat(sifat, exp_sifat, diffs)

    # Generate text explanations
    text_explanations = generate_sifat_explanation(sifat_table)

    # Combine sections (omit detailed table to keep output ringkas)
    html_output = f"""
    <div style="font-family: monospace; width: 100%;">
        <h3>Teks Rujukan (Utsmani)</h3>
        <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 24px;">
            {original_text}
        </div>

        <h3>Perbandingan Huruf</h3>
        {phoneme_html}
        
        <h3>Analisis Kesalahan Sifat Huruf</h3>
        <div style="padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #444;">
            {text_explanations}
        </div>
    </div>
    """

    return html_output


def explain_phonemes_html(dmp_obj, diffs):
    html_output = '<div style="padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 30px; color: inherit;">'

    # Process each difference (same logic as terminal version)
    for op, data in diffs:
        if op == dmp_obj.DIFF_EQUAL:
            html_output += f'<span style="color: currentColor;">{data}</span>'
        elif op == dmp_obj.DIFF_INSERT:
            html_output += f'<span style="color: #16a34a;">{data}</span>'
        elif op == dmp_obj.DIFF_DELETE:
            html_output += f'<span style="color: #c53030; text-decoration: line-through;">{data}</span>'

    html_output += "</div>"
    return html_output


def explain_sifat_html(table):
    if not table:
        return "<p>Tidak ada data sifat huruf yang tersedia</p>"

    # Kamus penjelasan untuk tooltip
    sifat_tooltips = {
        "hams": "Suara lembut tanpa dengung (seperti angin berhembus)",
        "jahr": "Suara kuat dengan getaran (seperti suara gemuruh)",
        "shadeed": "Bacaan kuat dan tegas (huruf harus ditekan)",
        "between": "Bacaan sedang antara kuat dan lembut",
        "rikhw": "Bacaan lembut dan ringan",
        "mofakham": "Bacaan tebal dan berat (seperti suara dalam)",
        "moraqaq": "Bacaan tipis dan ringan",
        "low_mofakham": "Bacaan agak tebal (sedikit berat)",
        "monfateh": "Mulut terbuka lebar saat membaca",
        "motbaq": "Mulut tertutup saat membaca",
        "safeer": "Ada desisan seperti siulan",
        "no_safeer": "Tidak ada desisan",
        "moqalqal": "Suara bergetar/bergoyang",
        "not_moqalqal": "Suara stabil tanpa getaran",
        "mokarar": "Ada pengulangan suara",
        "not_mokarar": "Tidak ada pengulangan",
        "motafashie": "Suara menyebar/meluas",
        "not_motafashie": "Suara tidak menyebar",
        "mostateel": "Bacaan panjang dan memanjang",
        "not_mostateel": "Bacaan pendek/tidak memanjang",
        "maghnoon": "Ada dengung (ghunnah)",
        "not_maghnoon": "Tidak ada dengung",
        # Tambahkan untuk sifat lainnya
        "tafkheem_or_taqeeq": "Penebalan (tafkheem) atau penipisan (tarqeeq) huruf",
        "idgham_or_izhar": "Peluluran (idgham) atau pengucapan jelas (izhar)",
        "sakt_or_idraj": "Jeda (sakt) atau tanpa jeda (idraj)",
        "wasl_or_waqf": "Bersambung (wasl) atau berhenti (waqf)",
        "ithbat_or_hadhf": "Penetapan (ithbat) atau penghapusan (hadhf)"
    }

    # Create HTML table with full width
    html_output = """
    <table style="width: 100%; border-collapse: collapse; background-color: #000; color: #fff; margin-bottom: 20px;">
        <thead>
            <tr>
    """

    # Get base columns (non-exp keys without 'tag')
    base_keys = [k for k in table[0].keys() if not k.startswith("exp_") and k != "tag"]

    # Add columns
    for key in base_keys:
        # Terjemahkan nama kolom ke bahasa Indonesia
        column_translations = {
            "phonemes": "Huruf",
            "hams": "Hams",
            "jahr": "Jahr", 
            "shadeed": "Shadeed",
            "between": "Antara",
            "rikhw": "Rikhw",
            "mofakham": "Mofakham",
            "moraqaq": "Moraqaq",
            "low_mofakham": "Low Mofakham",
            "monfateh": "Monfateh",
            "motbaq": "Motbaq",
            "safeer": "Safeer",
            "no_safeer": "No Safeer",
            "moqalqal": "Moqalqal",
            "not_moqalqal": "Not Moqalqal",
            "mokarar": "Mokarar",
            "not_mokarar": "Not Mokarar",
            "motafashie": "Motafashie",
            "not_motafashie": "Not Motafashie",
            "mostateel": "Mostateel",
            "not_mostateel": "Not Mostateel",
            "maghnoon": "Maghnoon",
            "not_maghnoon": "Not Maghnoon",
            # Tambahkan kolom untuk sifat lainnya
            "tafkheem_or_taqeeq": "Tafkheem/Tarqeeq",
            "idgham_or_izhar": "Idgham/Izhar",
            "sakt_or_idraj": "Sakt/Idraj", 
            "wasl_or_waqf": "Wasl/Waqf",
            "ithbat_or_hadhf": "Ithbat/Hadhf"
        }
        column_name = column_translations.get(key, key.replace("_", " ").title())
        
        # Tambahkan tooltip untuk kolom sifat
        tooltip = sifat_tooltips.get(key, "")
        if tooltip and key != "phonemes":
            html_output += f'<th style="border: 1px solid #444; padding: 8px; text-align: left; cursor: help;" title="{tooltip}">{column_name} ⓘ</th>'
        else:
            html_output += f'<th style="border: 1px solid #444; padding: 8px; text-align: left;">{column_name}</th>'

    html_output += """
            </tr>
        </thead>
        <tbody>
    """

    # Add rows
    for row in table:
        tag = row["tag"]
        html_output += "<tr>"

        for key in base_keys:
            exp_key = f"exp_{key}"
            value = str(row[key])

            # Terjemahkan nilai ke bahasa Indonesia jika bukan kolom phonemes
            if key != "phonemes":
                # Kamus terjemahan nilai sifat huruf
                value_translations = {
                    "hams": "همس",
                    "jahr": "جهر", 
                    "shadeed": "شديد",
                    "between": "بين الشدة والرخاوة",
                    "rikhw": "رخو",
                    "mofakham": "مفخم",
                    "moraqaq": "مرقق",
                    "low_mofakham": "أدنى المفخم",
                    "monfateh": "منفتح",
                    "motbaq": "مطبق",
                    "safeer": "صفير",
                    "no_safeer": "لا صفير",
                    "moqalqal": "مقلقل",
                    "not_moqalqal": "لا قلقلة",
                    "mokarar": "مكرر",
                    "not_mokarar": "لا تكرار",
                    "motafashie": "متفشي",
                    "not_motafashie": "لا تفشي",
                    "mostateel": "مستطيل",
                    "not_mostateel": "لا إستطالة",
                    "maghnoon": "مغن",
                    "not_maghnoon": "لا غنة",
                    # Tambahkan nilai untuk sifat lainnya
                    "tafkheem": "تفخيم", "tarqeeq": "ترقيق",
                    "idgham": "إدغام", "izhar": "إظهار", 
                    "sakt": "سكت", "idraj": "إدراج",
                    "wasl": "وصل", "waqf": "وقف",
                    "ithbat": "إثبات", "hadhf": "حذف",
                    "None": "-", "[PAD]": "-", "PAD": "-"
                }
                value = value_translations.get(value, value)

            # Apply styling based on tag and comparison
            if tag == "exact" and row.get(exp_key) != row[key]:
                html_output += f'<td style="border: 1px solid #444; padding: 8px; color: #ff0000;">{value}</td>'
            elif tag == "insert":
                html_output += f'<td style="border: 1px solid #444; padding: 8px; color: #ffff00;">{value}</td>'
            else:
                html_output += f'<td style="border: 1px solid #444; padding: 8px;">{value}</td>'

        html_output += "</tr>"

    html_output += """
        </tbody>
    </table>
    <div style="margin-top: 10px; color: #fff;">
        <strong>Keterangan Warna:</strong><br>
        <span style="color: #ff0000;">Merah</span>: Tidak sesuai dengan referensi<br>
        <span style="color: #ffff00;">Kuning</span>: Tambahan (tidak ada dalam referensi)<br>
        <span style="color: #ffffff;">Putih</span>: Sesuai dengan referensi<br><br>
        <strong>Tips:</strong> Arahkan kursor ke nama kolom (ⓘ) untuk melihat penjelasan cara membaca
    </div>
    """

    return html_output
