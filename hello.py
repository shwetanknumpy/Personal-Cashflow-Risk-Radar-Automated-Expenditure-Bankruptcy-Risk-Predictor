import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

class PersonalCashflowRiskRadar:
    def __init__(self, model, scaler):
        self.categories = ['Expense Mgmt', 'Debt Level', 'Liquidity', 'Savings', 'Overall']
        self.risk_data = []  # Store user data for analysis
        self.model = model
        self.scaler = scaler

    def collect_user_data(self):
        """
        Prompts user to input financial data. No hardcoded data used.
        """
        try:
            monthly_income = float(input("Enter your monthly income (INR): "))
            if monthly_income <= 0:
                raise ValueError("Income must be positive.")

            expenses = {}
            expense_categories = ['Rent/Housing', 'Food/Groceries', 'Transport', 'Entertainment/Misc']
            for cat in expense_categories:
                expenses[cat.lower()] = float(input(f"Enter monthly {cat} expense (INR): ")) or 0

            total_debts = float(input("Enter total debts (INR): ")) or 0
            total_savings = float(input("Enter total savings/assets (INR): ")) or 0
            cibil_score = int(input("Enter CIBIL score (300-900): "))
            if not (300 <= cibil_score <= 900):
                raise ValueError("CIBIL score must be between 300 and 900.")

            return monthly_income, expenses, total_debts, total_savings, cibil_score
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
            return self.collect_user_data()

    def calculate_risk_with_model(self, monthly_income, expenses, total_debts, total_savings):
        """
        Use trained Logistic Regression model to predict risk.
        """
        total_expenses = sum(expenses.values())

        # Features in the same order as model training
        features = [[
            total_expenses,                       # proxy for TransactionAmount
            (total_debts / (monthly_income * 12)) * 100,  # approx CreditUtilization %
            80,                                   # assume repayment history score, can extend to user input
            2,                                    # dummy NumLoans
            5,                                    # dummy CreditAge
            1                                     # dummy NewLoans
        ]]

        features_scaled = self.scaler.transform(features)
        risk_prediction = self.model.predict(features_scaled)[0]
        risk_prob = self.model.predict_proba(features_scaled)[0][1] * 100

        return risk_prediction, risk_prob

    def check_cibil(self, cibil_score):
        if cibil_score >= 800:
            category = "Excellent"
            insights = "Excellent credit health. Low risk of rejection."
        elif 650 <= cibil_score < 800:
            category = "Good"
            insights = "Good credit. Eligible for most loans."
        elif 550 <= cibil_score < 650:
            category = "Average"
            insights = "Average credit. May face higher interest."
        else:
            category = "Poor"
            insights = "Poor credit. High risk."
        return category, insights

    def plot_histogram_and_scatter(self):
        """
        Use Matplotlib and Seaborn for histograms and scatter plots to demonstrate.
        """
        if not self.risk_data:
            print("No data to plot. Collect user data first.")
            return

        df = pd.DataFrame(self.risk_data)

        # Histogram of overall risk
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        sns.histplot(df['risk_prob'], bins=10, kde=True, color='blue')
        plt.title("Histogram of Predicted Risk Probabilities")
        plt.xlabel("Risk Probability (%)")

        # Scatter plot: Expenses vs Risk
        plt.subplot(1, 3, 2)
        sns.scatterplot(data=df, x='expenses', y='risk_prob', hue='savings', palette='viridis')
        plt.title("Scatter Plot: Expenses vs Risk Probability")
        plt.xlabel("Monthly Expenses (INR)")
        plt.ylabel("Risk Probability (%)")

        # Bar chart of average expenses vs debts
        plt.subplot(1, 3, 3)
        avg_expenses = df['expenses'].mean()
        avg_debts = df['debts'].mean()
        data = [avg_expenses, avg_debts]
        plt.bar(['Avg Expenses', 'Avg Debts'], data, color=['green', 'red'])
        plt.title("Average Expenses and Debts")
        plt.ylabel("INR")

        plt.tight_layout()
        plt.savefig("risk_demo_plots.png")
        plt.show()

    def run(self):
        """
        Main loop to collect data, calculate, and demonstrate.
        """
        while True:
            print("\n--- Personal Cashflow Risk Radar ---")
            monthly_income, expenses, total_debts, total_savings, cibil_score = self.collect_user_data()

            risk_prediction, risk_prob = self.calculate_risk_with_model(monthly_income, expenses, total_debts, total_savings)
            category, insights = self.check_cibil(cibil_score)

            print("\n**Risk Prediction:**", "High Risk" if risk_prediction == 1 else "Low Risk")
            print(f"**Predicted Risk Probability:** {risk_prob:.2f}%")
            print(f"**CIBIL Category:** {category}")
            print(f"**CIBIL Insights:** {insights}")

            # Store for plotting
            self.risk_data.append({
                'income': monthly_income,
                'expenses': sum(expenses.values()),
                'debts': total_debts,
                'savings': total_savings,
                'risk_prob': risk_prob
            })

            self.plot_histogram_and_scatter()

            more = input("Analyze another user? (y/n): ").lower()
            if more != 'y':
                break


# -------------------- Training Part --------------------
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("transactions_dataset.csv")

    # Create Risk label (1 = risky, 0 = safe)
    df["Risk"] = ((df["CreditUtilization"] > 70) | (df["RepaymentHistory"] < 60)).astype(int)

    # Features & Target
    X = df[["TransactionAmount", "CreditUtilization", "RepaymentHistory", "NumLoans", "CreditAge", "NewLoans"]]
    y = df["Risk"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    print("✅ Model trained successfully!")
    print("Classification Report:\n", classification_report(y_test, model.predict(X_test_scaled)))
    print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test_scaled)))

    # Run interactive system
    radar = PersonalCashflowRiskRadar(model, scaler)
    radar.run()

