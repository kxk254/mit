
def convert_dxf_to_shift_jis(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            content = infile.read()

        with open(output_file_path, 'w', encoding='shift_jis', errors='ignore') as outfile:
            outfile.write(content)

        print(f"変換完了: {output_file_path} に Shift-JIS 形式で保存しました。")

    except UnicodeDecodeError:
        print("⚠ エンコーディングエラー：UTF-8 で読み取れませんでした。")
    except Exception as e:
        print(f"⚠ エラーが発生しました: {e}")


# === 実行例 ===
# 必要に応じてファイルパスを変更してください
if __name__ == "__main__":
    input_path = "C:\\Users\\konno\\OneDrive - SCM\\2_Soliton_Deals\\02. CS Holdings\\02. Real Estate\\01. 築地\\リーシング\\９F 11F\\tcb-9f11f\\CADデータ11F - コピー\\竣工図_内装施工図-展開図.dxf"      # 元のDXFファイル
    output_path = "C:\\Users\\konno\\OneDrive - SCM\\2_Soliton_Deals\\02. CS Holdings\\02. Real Estate\\01. 築地\\リーシング\\９F 11F\\tcb-9f11f\\CADデータ11F - コピー\\竣工図_内装施工図-展開図sjis.dxf"  # Shift-JISで保存するファイル
    convert_dxf_to_shift_jis(input_path, output_path)