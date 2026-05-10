import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, ttest_ind
from scikit_posthocs import posthoc_nemenyi_friedman
import argparse

def perform_bonferroni_dunn(df, baseline_column, test_type):
    baseline_values = df[baseline_column]
    bonferroni_results = {}
    
    for column in df.columns:
        if column != baseline_column:
            stat, p = ttest_ind(baseline_values, df[column])
            # Bonferroni correction
            p_corrected = p * (len(df.columns) - 1)
            bonferroni_results[f'{baseline_column} vs {column}'] = p_corrected

    print(f"Bonferroni-Dunn Test Results for {test_type}:")
    for comparison, p_corrected in bonferroni_results.items():
        print(f"{comparison}: p-value (Bonferroni corrected) = {p_corrected:.4f}")

def perform_nemenyi(df, test_type):
    print(f"Performing Nemenyi Test for {test_type}...")
    nemenyi_results = posthoc_nemenyi_friedman(df)
    print(nemenyi_results)

def perform_friedman(df, test_type):
    print(f"Performing Friedman Test for {test_type}...")
    friedman_stat, friedman_p = friedmanchisquare(*[df[col] for col in df.columns])
    print(f"Friedman Test Statistic: {friedman_stat:.4f}, p-value: {friedman_p:.4f}")

def perform_wilcoxon_tests(alpha_file, epsilon_file, output_file):
    # Load the CSV files
    df_alpha = pd.read_csv(alpha_file, index_col=0)
    df_epsilon = pd.read_csv(epsilon_file, index_col=0)
    
    # Initialize dictionaries to store Wilcoxon test results for both alpha and epsilon
    wilcoxon_results_alpha = {}
    wilcoxon_results_epsilon = {}

    # Perform Wilcoxon tests for Alpha (diagnostic accuracy)
    for i in range(len(df_alpha.columns)):
        for j in range(i + 1, len(df_alpha.columns)):
            model_1 = df_alpha.columns[i]
            model_2 = df_alpha.columns[j]
            stat, p_value = wilcoxon(df_alpha[model_1], df_alpha[model_2])
            wilcoxon_results_alpha[f"{model_1} vs {model_2}"] = {"statistic": stat, "p-value": p_value}

    # Perform Wilcoxon tests for Epsilon (error rate)
    for i in range(len(df_epsilon.columns)):
        for j in range(i + 1, len(df_epsilon.columns)):
            model_1 = df_epsilon.columns[i]
            model_2 = df_epsilon.columns[j]
            stat, p_value = wilcoxon(df_epsilon[model_1], df_epsilon[model_2])
            wilcoxon_results_epsilon[f"{model_1} vs {model_2}"] = {"statistic": stat, "p-value": p_value}

    # Save the results to a file
    with open(output_file, 'w') as f:
        f.write("Wilcoxon Test Results for Alpha (Diagnostic Accuracy)\n")
        for comparison, result in wilcoxon_results_alpha.items():
            f.write(f"{comparison}: Statistic = {result['statistic']}, p-value = {result['p-value']}\n")
        
        f.write("\nWilcoxon Test Results for Epsilon (Error Rate)\n")
        for comparison, result in wilcoxon_results_epsilon.items():
            f.write(f"{comparison}: Statistic = {result['statistic']}, p-value = {result['p-value']}\n")
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Perform statistical tests (Bonferroni-Dunn, Nemenyi, Friedman, Wilcoxon) on datasets")
    
    # Define command-line arguments
    parser.add_argument("beta_file", type=str, nargs="?", help="Path to the CSV file containing the Beta dataset")
    parser.add_argument("cbar_file", type=str, nargs="?", help="Path to the CSV file containing the C-bar dataset")
    parser.add_argument("baseline_beta", type=str, nargs="?", help="Baseline column name for Beta values")
    parser.add_argument("baseline_cbar", type=str, nargs="?", help="Baseline column name for C-bar values")
    
    parser.add_argument("--alpha", type=str, help="Path to the CSV file containing Alpha values (for Wilcoxon test)")
    parser.add_argument("--epsilon", type=str, help="Path to the CSV file containing Epsilon values (for Wilcoxon test)")
    parser.add_argument("--output", type=str, help="Path to save the Wilcoxon test results")
    
    parser.add_argument("--test", choices=["bonferroni", "nemenyi", "friedman", "wilcoxon"], default="bonferroni", help="Type of test to perform")

    args = parser.parse_args()

    # Perform the selected test
    if args.test == "bonferroni":
        perform_bonferroni_dunn(pd.read_csv(args.beta_file), args.baseline_beta, "Beta Values")
        perform_bonferroni_dunn(pd.read_csv(args.cbar_file), args.baseline_cbar, "C-bar Values")
    elif args.test == "nemenyi":
        perform_nemenyi(pd.read_csv(args.beta_file), "Beta Values")
        perform_nemenyi(pd.read_csv(args.cbar_file), "C-bar Values")
    elif args.test == "friedman":
        perform_friedman(pd.read_csv(args.beta_file), "Beta Values")
        perform_friedman(pd.read_csv(args.cbar_file), "C-bar Values")
    elif args.test == "wilcoxon":
        if not args.alpha or not args.epsilon or not args.output:
            print("For Wilcoxon test, please provide --alpha, --epsilon, and --output files")
        else:
            perform_wilcoxon_tests(args.alpha, args.epsilon, args.output)

if __name__ == "__main__":
    main()
