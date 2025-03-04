import sys
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                            QPushButton, QCheckBox, QFrame, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator, QFont
from PyQt6.QtPrintSupport import QPrinter, QPrintDialog

class PCRCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set Consolas font for the entire application
        app_font = QFont("Consolas", 10)
        QApplication.setFont(app_font)
        
        self.setWindowTitle("PCR MASTER-MIX Calculator")
        self.setFixedSize(800, 500)
        
        # Constants for spacing and padding
        self.SPACING = 10
        self.PADDING = 20
        self.COLUMN_WIDTH = 150
        self.INPUT_HEIGHT = 30
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(self.SPACING)
        layout.setContentsMargins(self.PADDING, self.PADDING, self.PADDING, self.PADDING)
        
        # Top section
        top_layout = QHBoxLayout()
        top_layout.setSpacing(self.SPACING)
        
        # Create and style input fields
        self.volume = self._create_input("20")
        self.reactions = self._create_input("1")
        
        top_layout.addWidget(self._create_label("VOL. PER RXN (µl):"))
        top_layout.addWidget(self.volume)
        top_layout.addWidget(self._create_label("#REACTIONS:"))
        top_layout.addWidget(self.reactions)
        
        self.mg_buff_check = QCheckBox("1.5mM MgCl₂ BUFFER")
        self.mg_buff_check.setFixedHeight(self.INPUT_HEIGHT)
        top_layout.addWidget(self.mg_buff_check)
        
        # Add print button
        self.print_button = QPushButton("Print")
        self.print_button.setFixedHeight(self.INPUT_HEIGHT)
        self.print_button.clicked.connect(self.print_results)
        top_layout.addWidget(self.print_button)
        
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # Grid setup
        grid = QGridLayout()
        grid.setSpacing(self.SPACING)
        
        # Headers
        headers = ["~", "STOCK CONC.", "FINAL CONC.", "1 RXN (µl)", "MASTER MIX (µl)"]
        for i, header in enumerate(headers):
            label = self._create_header_label(header)
            grid.addWidget(label, 0, i)
        
        self.stock_inputs = {}
        self.desired_inputs = {}
        self.rxn_outputs = {}
        self.mm_outputs = {}
        
        # Ingredients definition
        self.ingredients = [
            ("Buffer", "10", "1", "X"),
            ("MgCl₂", "25", "2.5", "mM"),
            ("dNTP", "10", "0.2", "mM"),
            ("Fwd Primer", "10", "0.2", "µM"),
            ("Rev Primer", "10", "0.2", "µM"),
            ("Taq", "5", "0.5", ["U/µl", "U/rxn"]), 
            ("DNA", "", "1", "µl"),
            ("H₂O", "", "", "")
        ]
        
        # Create grid rows
        for row, (name, stock, desired, unit) in enumerate(self.ingredients, 1):
            grid.addWidget(self._create_label(name), row, 0)
            
            # Stock concentration
            stock_widget = QWidget()
            stock_layout = QHBoxLayout(stock_widget)
            stock_layout.setContentsMargins(0, 0, 0, 0)
            stock_layout.setSpacing(5)
            
            if stock:
                stock_input = self._create_input(stock)
                self.stock_inputs[name] = stock_input
                stock_layout.addWidget(stock_input)
                if unit:
                    if isinstance(unit, list) and name == "Taq":
                        unit_label = self._create_label(unit[0])  # U/µl for stock
                    elif not isinstance(unit, list):
                        unit_label = self._create_label(unit)
                    unit_label.setFixedWidth(40)
                    stock_layout.addWidget(unit_label)
            grid.addWidget(stock_widget, row, 1)
            
            # Final concentration
            desired_widget = QWidget()
            desired_layout = QHBoxLayout(desired_widget)
            desired_layout.setContentsMargins(0, 0, 0, 0)
            desired_layout.setSpacing(5)
            
            if desired or name == "DNA":
                desired_input = self._create_input(desired)
                self.desired_inputs[name] = desired_input
                desired_layout.addWidget(desired_input)
                if unit:
                    if isinstance(unit, list) and name == "Taq":
                        unit_label = self._create_label(unit[1])  # U/rxn for desired
                    elif not isinstance(unit, list):
                        unit_label = self._create_label(unit)
                    unit_label.setFixedWidth(40)
                    desired_layout.addWidget(unit_label)
            grid.addWidget(desired_widget, row, 2)
            
            # Output fields
            self.rxn_outputs[name] = self._create_output()
            self.mm_outputs[name] = self._create_output()
            grid.addWidget(self.rxn_outputs[name], row, 3)
            grid.addWidget(self.mm_outputs[name], row, 4)
        
        # Total row
        total_row = len(self.ingredients) + 1
        grid.addWidget(self._create_label("TOTAL VOL."), total_row, 0)
        self.total_rxn = self._create_output()
        self.total_mm = self._create_output()
        self.total_rxn.setStyleSheet("font-weight: bold; font-family: Consolas")
        self.total_mm.setStyleSheet("font-weight: bold; font-family: Consolas")
        grid.addWidget(self.total_rxn, total_row, 3)
        grid.addWidget(self.total_mm, total_row, 4)
        
        # Set column widths
        for col in range(5):
            grid.setColumnMinimumWidth(col, self.COLUMN_WIDTH)
        
        layout.addLayout(grid)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Calculate button
        calc_button = QPushButton("CALCULATE")
        calc_button.setFixedHeight(self.INPUT_HEIGHT)
        calc_button.clicked.connect(self.calculate)
        layout.addWidget(calc_button)
        
        # Connect signals
        self.volume.textChanged.connect(self.calculate)
        self.reactions.textChanged.connect(self.calculate)
        self.mg_buff_check.stateChanged.connect(self.calculate)
        
        self.calculate()
    
    def _create_input(self, default_value=""):
        input_field = QLineEdit(default_value)
        input_field.setValidator(QDoubleValidator(0.0, 1000.0, 3))
        input_field.setAlignment(Qt.AlignmentFlag.AlignRight)
        input_field.setFixedHeight(self.INPUT_HEIGHT)
        return input_field
    
    def _create_output(self):
        output_field = QLineEdit()
        output_field.setReadOnly(True)
        output_field.setAlignment(Qt.AlignmentFlag.AlignRight)
        output_field.setFixedHeight(self.INPUT_HEIGHT)
        return output_field
    
    def _create_label(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setFixedHeight(self.INPUT_HEIGHT)
        return label
    
    def _create_header_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedHeight(self.INPUT_HEIGHT)
        return label

    def print_results(self):
        try:
            # Get current datetime for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"pcr_mm_calc_{timestamp}.txt"
            
            # Open file dialog for save location
            from PyQt6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save PCR Results",
                str(Path.home() / default_filename),
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                # Generate and write results to file
                text = self.generate_print_text()
                Path(file_path).write_text(text, encoding='utf-8')
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Results saved to:\n{file_path}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save results: {str(e)}"
            )


    def generate_print_text(self):
        output = []
        output.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        output.append("")
        output.append("PCR master-mix calculation")
        output.append("=" * 50)
        output.append("")
        output.append(f"Volume per reaction: {self.volume.text()} µl")
        output.append(f"Number of reactions: {self.reactions.text()}")
        output.append(f"Using 1.5mM MgCl₂ buffer: {'Yes' if self.mg_buff_check.isChecked() else 'No'}")
        output.append("")
        
        # Table header
        output.append(f"{'Component':<15} {'Stock':<10} {'Final':<10} {'1 Rxn':<10} {'Master Mix':<10}")
        output.append("-" * 60)
        
        # Table content
        for name, _, _, _ in self.ingredients:
            stock = self.stock_inputs.get(name, QLineEdit()).text()
            final = self.desired_inputs.get(name, QLineEdit()).text()
            rxn = self.rxn_outputs[name].text()
            mm = self.mm_outputs[name].text()
            output.append(f"{name:<15} {stock:<10} {final:<10} {rxn:<10} {mm:<10}")
        
        # Total
        output.append("-" * 60)
        output.append(f"{'TOTAL VOL.':<15} {'':<10} {'':<10} {self.total_rxn.text():<10} {self.total_mm.text():<10}")
        
        return "\n".join(output)

    def calculate(self):
        try:
            vol = float(self.volume.text())
            rxn = float(self.reactions.text())
            total_volume = 0
            total_mm_volume = 0
            
            # Calculate volumes for all ingredients except water
            for name, stock, desired, _ in self.ingredients:
                if name == "H₂O":
                    continue
                
                if name == "DNA":
                    amount = float(self.desired_inputs[name].text())
                    total_volume += amount
                
                else:
                    stock_conc = float(self.stock_inputs[name].text())
                    final_conc = float(self.desired_inputs[name].text())
                    
                    if name == "MgCl₂" and self.mg_buff_check.isChecked():
                        final_conc -= 1.5
                    
                    
                    if name == "Taq":
                        amount = final_conc / stock_conc
                    else:
                        amount = (final_conc * vol) / stock_conc
                    
                    total_volume += amount
                
                self.rxn_outputs[name].setText(f"{amount:.2f}")
                self.mm_outputs[name].setText(f"{amount * rxn:.2f}")
                total_mm_volume += amount * rxn
            
            # Calculate water
            water_volume = vol - total_volume
            total_volume += water_volume
            total_mm_volume += water_volume * rxn
            
            self.rxn_outputs["H₂O"].setText(f"{water_volume:.2f}")
            self.mm_outputs["H₂O"].setText(f"{water_volume * rxn:.2f}")
            
            # Update total volume
            self.total_rxn.setText(f"{total_volume:.2f}")
            self.total_mm.setText(f"{total_mm_volume:.2f}")
            
        except ValueError:
            for name, _, _, _ in self.ingredients:
                self.rxn_outputs[name].setText("")
                self.mm_outputs[name].setText("")
            self.total_rxn.setText("")
            self.total_mm.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PCRCalculator()
    window.show()
    sys.exit(app.exec())
