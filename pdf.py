# Create a PDF summary using fpdf library
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Baseball Game Simulation Code Summary", ln=True, align="C")
        self.ln(10)

    def chapter_body(self, text):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, text)
        self.ln()

# Summary text
summary_text = (
    "This Python code simulates baseball innings using probabilistic outcomes based on real statistical data. \n\n"
    "What the Code Does:\n"
    "- It defines an enumeration of possible events in a plate appearance, such as walk, single, double, triple, home run, double play, out, and strikeout.\n"
    "- The 'inning' function simulates one inning by processing plate appearances for each batter, updating base runner positions, and tracking outs and runs scored.\n"
    "- It incorporates statistical data (e.g., walk rate, home run rate) with multipliers for each batting order position, adding realism to the simulation.\n"
    "- Special cases like double plays and situational advances on outs are also handled.\n\n"
    "Flexibility:\n"
    "- The use of adjustable coefficient arrays makes it easy to modify probabilities for different players or teams.\n"
    "- The modular design of the inning simulation allows for integration into larger game simulations or for adapting to different game scenarios.\n"
    "- Running the simulation over a large number of innings produces reliable averages for game analysis.\n\n"
    "Applications:\n"
    "- Performance Analysis: Assessing how different batting orders or player statistics influence run production.\n"
    "- Game Strategy and Coaching: Evaluating the impact of lineup changes or in-game decisions using simulated outcomes.\n"
    "- Educational and Research Purposes: Demonstrating the application of probability and simulation techniques in sports analytics.\n\n"
    "Overall, this code serves as a robust tool for exploring various scenarios in baseball game analysis and can be tailored to simulate different conditions or team profiles."
)

pdf = PDF()
pdf.add_page()
pdf.chapter_body(summary_text)
pdf.output("baseball_simulation_summary.pdf")
