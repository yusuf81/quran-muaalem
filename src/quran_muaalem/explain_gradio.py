from typing import Literal
import diff_match_patch as dmp

from .explain import expalin_sifat
from .modeling.vocab import SIFAT_ATTR_TO_ARABIC_WITHOUT_BRACKETS


def generate_sifat_explanation(table, lang):
    """Generate Indonesian text explanations for sifat comparison results"""
    if not table:
        return ""
    
    explanations = []
    
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
                        "maghnoon": "sifat maghnoon", "not_maghnoon": "sifat not maghnoon"
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
                        "None": "-"
                    }
                    
                    sifat_name = sifat_names.get(base_key, base_key)
                    actual_translated = value_translations.get(actual, actual)
                    expected_translated = value_translations.get(expected, expected)
                    
                    explanations.append(
                        f"❌ <strong>Huruf '{phoneme}'</strong>: {sifat_name} tidak sesuai. "
                        f"Dibaca: <span style='color: #ff0000;'>{actual_translated}</span>, "
                        f"Seharusnya: <span style='color: #00ff00;'>{expected_translated}</span>"
                    )
        
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
    lang: Literal["arabic", "english"] = "english",
) -> str:
    # Create diff-match-patch object
    dmp_obj = dmp.diff_match_patch()

    # Calculate differences using Google's diff-match-patch (same as terminal)
    diffs = dmp_obj.diff_main(exp_phonemes, phonemes)

    # Create HTML for phoneme differences
    phoneme_html = explain_phonemes_html(dmp_obj, diffs)

    # Create HTML for sifat table using your existing function
    sifat_table = expalin_sifat(sifat, exp_sifat, diffs)
    sifat_html = explain_sifat_html(sifat_table, lang)
    
    # Generate text explanations
    text_explanations = generate_sifat_explanation(sifat_table, lang)

    # Combine all sections
    html_output = f"""
    <div style="font-family: monospace; width: 100%;">
        <h3>Perbandingan Huruf</h3>
        {phoneme_html}
        
        <h3>Analisis Kesalahan Sifat Huruf</h3>
        <div style="background-color: #1a1a1a; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #fff;">
            {text_explanations}
        </div>
        
        <h3>Detail Perbandingan Sifat Huruf</h3>
        {sifat_html}
    </div>
    """

    return html_output


def explain_phonemes_html(dmp_obj, diffs):
    html_output = '<div style="background-color: #000; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 30px;">'

    # Process each difference (same logic as terminal version)
    for op, data in diffs:
        if op == dmp_obj.DIFF_EQUAL:
            html_output += f'<span style="color: #ffffff;">{data}</span>'
        elif op == dmp_obj.DIFF_INSERT:
            html_output += f'<span style="color: #00ff00;">{data}</span>'
        elif op == dmp_obj.DIFF_DELETE:
            html_output += f'<span style="color: #ff0000; text-decoration: line-through;">{data}</span>'

    html_output += "</div>"
    return html_output


def explain_sifat_html(table, lang):
    if not table:
        return "<p>Tidak ada data sifat huruf yang tersedia</p>"

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
            "not_maghnoon": "Not Maghnoon"
        }
        column_name = column_translations.get(key, key.replace("_", " ").title())
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
                    "None": "-"
                }
                value = value_translations.get(value, value)

            # Apply styling based on tag and comparison
            if tag == "exact" and row.get(exp_key) != row[key]:
                html_output += f'<td style="border: 1px solid #444; padding: 8px; color: #ff0000;">{value}</td>'
            elif tag == "insert":
                html_output += f'<td style="border: 1px solid #444; padding: 8px; color: #ffff00;">{value}</td>'
            else:
                html_output += (
                    f'<td style="border: 1px solid #444; padding: 8px;">{value}</td>'
                )

        html_output += "</tr>"

    html_output += """
        </tbody>
    </table>
    <div style="margin-top: 10px; color: #fff;">
        <strong>Keterangan Warna:</strong><br>
        <span style="color: #ff0000;">Merah</span>: Tidak sesuai dengan referensi<br>
        <span style="color: #ffff00;">Kuning</span>: Tambahan (tidak ada dalam referensi)<br>
        <span style="color: #ffffff;">Putih</span>: Sesuai dengan referensi
    </div>
    """

    return html_output
