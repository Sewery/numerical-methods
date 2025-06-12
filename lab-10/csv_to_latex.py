import csv
import math

def csv_to_latex_table(csv_file, caption, label, skip_rows=0):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        
        for _ in range(skip_rows):
            next(reader)
        
        header = next(reader)
        
        data = list(reader)
    
    latex = "\\begin{table}[H]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    
    num_columns = len(header)
    latex += "\\begin{tabular}{|" + "c|" * num_columns + "}\n"
    latex += "\\hline\n"
    
    latex += f"${header[0]}$ & " + " & ".join([f"${col}$" for col in header[1:]]) + " \\\\\n"
    latex += "\\hline\n"
    
    for row in data:
        n = row[0]
        values = []
        
        for val in row[1:]:
            try:
                float_val = float(val)
                if float_val == 0:
                    values.append("$0.00 \\times 10^{0}$")
                else:
                    exponent = math.floor(math.log10(abs(float_val)))
                    mantissa = float_val / (10 ** exponent)
                    values.append(f"${mantissa:.2f} \\times 10^{{{exponent}}}$")
            except ValueError:
                values.append(val)
        
        latex += f"{n} & " + " & ".join(values) + " \\\\\n"
    
    # Kończymy tabelę
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"
    return latex

# Przykładowe użycie
csv_file = r".\results\csv\iterations_residual.csv"
caption = "Ilość iteracji dla kryterium residualnego w zależności od rozmiaru macierzy"
label = "tab:iterations_residual"

# Pomijamy pierwszy wiersz "Dokladnosc ro"
latex_code = csv_to_latex_table(csv_file, caption, label, skip_rows=1)
print(latex_code)