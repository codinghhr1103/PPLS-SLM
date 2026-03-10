"""
Results Visualization for PPLS Parameter Estimation
==================================================

This module creates publication-ready plots for parameter comparison and results analysis,
including the loading matrix comparisons shown in Figures 2-3 of the paper. Uses a simplified
directory structure reading from base_dir and saving figures to figures directory.

Architecture Overview:
---------------------
The module provides:
1. PPLSVisualizer: Main visualization coordinator
2. LoadingPlotter: Specialized plotting for loading matrices (sine function comparisons)
3. PerformancePlotter: Visualization of algorithm performance metrics

Function List:
--------------
PPLSVisualizer:
    - __init__(base_dir, figure_format): Initialize with settings
    - plot_loading_comparison(trial_result, component_idx): Generate Figures 2-3 style plots
    - plot_parameter_recovery(analysis_results): Visualize recovery across trials
    - plot_convergence_history(trial_results): Plot algorithm convergence
    - create_results_summary(experiment_results): Generate comprehensive dashboard
    - save_all_figures(): Save all generated figures

LoadingPlotter:
    - plot_sine_comparison(W_true, W_slm, W_em, W_ecm, C_true, C_slm, C_em, C_ecm): Compare loadings
    - plot_component_wise(W_list, C_list, labels): Plot individual components
    - add_statistical_bands(ax, data, color): Add confidence/prediction bands
    - customize_axes(ax, title, xlabel, ylabel): Setup axes formatting
    - create_loading_heatmap(W, C, title): Heatmap visualization of loadings

PerformancePlotter:
    - plot_mse_comparison(slm_metrics, em_metrics, ecm_metrics): Box plots of MSE
    - plot_bias_variance_tradeoff(metrics_dict): Bias-variance decomposition
    - plot_computational_time(timing_data): Runtime comparison
    - create_correlation_heatmap(correlations): Parameter correlation heatmap
    - plot_recovery_rates(recovery_data): Success rate visualization

Call Relationships:
------------------
PPLSVisualizer.plot_loading_comparison() → LoadingPlotter.plot_sine_comparison()
PPLSVisualizer.plot_parameter_recovery() → PerformancePlotter.plot_mse_comparison()
PPLSVisualizer.create_results_summary() → LoadingPlotter.create_loading_heatmap()
PPLSVisualizer.create_results_summary() → PerformancePlotter.plot_recovery_rates()
LoadingPlotter.plot_sine_comparison() → LoadingPlotter.customize_axes()
PerformancePlotter.plot_mse_comparison() → matplotlib.pyplot functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import json
import pickle


class PPLSVisualizer:
    """
    Main visualization coordinator for PPLS experiment results.
    Creates publication-ready figures for parameter estimation comparison.
    """
    
    def __init__(self, base_dir: str, figure_format: str = 'pdf'):
        """
        Initialize visualizer with base directory and settings.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing data and results
        figure_format : str
            Output format for figures ('pdf', 'png', 'svg')
        """
        self.base_dir = base_dir
        self.figure_format = figure_format
        self.figure_dir = os.path.join(base_dir, 'figures')
        self.data_dir = os.path.join(base_dir, 'data')
        self.results_dir = os.path.join(base_dir, 'results')
        
        # Create figures directory
        os.makedirs(self.figure_dir, exist_ok=True)
        
        # Set publication-quality plotting style  
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['legend.framealpha'] = 0.9
        
        # Initialize plotters
        self.loading_plotter = LoadingPlotter()
        self.performance_plotter = PerformancePlotter()
        
    def plot_loading_comparison(self, trial_result: Dict, component_idx: int = 0):
        """
        Generate loading matrix comparison plots for all three algorithms (Figures 2-3 style).
        
        Parameters:
        -----------
        trial_result : dict
            Trial result containing true and estimated parameters
        component_idx : int
            Which component to plot (default: first component)
        """
        # Extract parameters
        true_params = trial_result['true_params']
        slm_results = trial_result['slm_results']
        bcd_slm_results = trial_result.get('bcd_slm_results')
        slm_oracle_results = trial_result.get('slm_oracle_results')
        em_results = trial_result['em_results']
        ecm_results = trial_result['ecm_results']


        W_bcd = None
        C_bcd = None
        if isinstance(bcd_slm_results, dict) and bcd_slm_results:
            W_bcd = bcd_slm_results.get('W')
            C_bcd = bcd_slm_results.get('C')

        W_slm_oracle = None
        C_slm_oracle = None
        if isinstance(slm_oracle_results, dict) and slm_oracle_results:
            W_slm_oracle = slm_oracle_results.get('W')
            C_slm_oracle = slm_oracle_results.get('C')

        
        # Create separate figures for W and C
        fig_W = self.loading_plotter.plot_W_comparison(
            true_params['W'],
            slm_results['W'],
            em_results['W'],
            ecm_results['W'],
            component_idx,
            W_bcd=W_bcd,
            W_slm_oracle=W_slm_oracle,
        )

        
        fig_C = self.loading_plotter.plot_C_comparison(
            true_params['C'],
            slm_results['C'],
            em_results['C'],
            ecm_results['C'],
            component_idx,
            C_bcd=C_bcd,
            C_slm_oracle=C_slm_oracle,
        )

        
        # Save figures with publication-quality naming
        filename_W = f'loading_W_Component_{component_idx+1}.{self.figure_format}'
        fig_W.savefig(os.path.join(self.figure_dir, filename_W), 
                     bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig_W)
        
        filename_C = f'loading_C_Component_{component_idx+1}.{self.figure_format}'
        fig_C.savefig(os.path.join(self.figure_dir, filename_C), 
                     bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig_C)
        
    def plot_parameter_recovery(self, analysis_results: Dict):
        """
        Visualize parameter recovery quality across all trials.
        
        Parameters:
        -----------
        analysis_results : dict
            Analysis results from experiment
        """
        # Extract metrics for all three algorithms
        slm_metrics = analysis_results.get('slm', {})
        bcd_metrics = analysis_results.get('bcd_slm', {})
        em_metrics = analysis_results.get('em', {})
        ecm_metrics = analysis_results.get('ecm', {})
        
        # Create MSE comparison bar plot
        fig = self.performance_plotter.plot_mse_bars(slm_metrics, bcd_metrics, em_metrics, ecm_metrics)

        
        filename = f'Figure_4_MSE_Comparison.{self.figure_format}'
        fig.savefig(os.path.join(self.figure_dir, filename), 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig)
        
        # Export MSE comparison to Excel
        self._export_mse_to_excel(slm_metrics, bcd_metrics, em_metrics, ecm_metrics)

            
    def plot_convergence_history(self, trial_results: List[Dict]):
        """
        Export convergence statistics to Excel table instead of plot.
        
        Parameters:
        -----------
        trial_results : List[Dict]
            Results from all trials
        """
        # Extract convergence data (iterations are summarised over successful trials only).
        slm_iterations: List[int] = []
        bcd_slm_iterations: List[int] = []
        slm_joint_iterations: List[int] = []
        slm_oracle_iterations: List[int] = []
        em_iterations: List[int] = []
        ecm_iterations: List[int] = []

        slm_success: List[int] = []
        bcd_slm_success: List[int] = []
        slm_joint_success: List[int] = []
        slm_oracle_success: List[int] = []
        em_success: List[int] = []
        ecm_success: List[int] = []



        for trial in trial_results:
            slm_res = trial.get('slm_results', {})
            slm_ok = bool(slm_res.get('success', False))
            slm_success.append(int(slm_ok))
            if slm_ok and 'n_iterations' in slm_res:
                slm_iterations.append(int(slm_res['n_iterations']))

            bcd_res = trial.get('bcd_slm_results', {})
            bcd_ok = bool(bcd_res.get('success', False))
            bcd_slm_success.append(int(bcd_ok))
            if bcd_ok and 'n_iterations' in bcd_res:
                bcd_slm_iterations.append(int(bcd_res['n_iterations']))

            slm_joint_res = trial.get('slm_joint_results', {})

            slm_joint_ok = bool(slm_joint_res.get('success', False))
            slm_joint_success.append(int(slm_joint_ok))
            if slm_joint_ok and 'n_iterations' in slm_joint_res:
                slm_joint_iterations.append(int(slm_joint_res['n_iterations']))

            slm_or_res = trial.get('slm_oracle_results', {})
            slm_or_ok = bool(slm_or_res.get('success', False))
            slm_oracle_success.append(int(slm_or_ok))
            if slm_or_ok and 'n_iterations' in slm_or_res:
                slm_oracle_iterations.append(int(slm_or_res['n_iterations']))

            em_res = trial.get('em_results', {})
            em_ok = bool(em_res.get('log_likelihood', -np.inf) > -np.inf)
            em_success.append(int(em_ok))
            if em_ok and 'n_iterations' in em_res:
                em_iterations.append(int(em_res['n_iterations']))

            ecm_res = trial.get('ecm_results', {})
            ecm_ok = bool(ecm_res.get('log_likelihood', -np.inf) > -np.inf)
            ecm_success.append(int(ecm_ok))
            if ecm_ok and 'n_iterations' in ecm_res:
                ecm_iterations.append(int(ecm_res['n_iterations']))


        # Export convergence comparison table to Excel
        self._export_convergence_table_to_excel(
            slm_iterations,
            bcd_slm_iterations,
            slm_joint_iterations,
            slm_oracle_iterations,
            em_iterations,
            ecm_iterations,
            success_rates={
                'SLM-Fixed': float(np.mean(slm_success)) if slm_success else 0.0,
                'BCD-SLM': float(np.mean(bcd_slm_success)) if bcd_slm_success else 0.0,
                'SLM-Joint': float(np.mean(slm_joint_success)) if slm_joint_success else 0.0,
                'SLM-Oracle': float(np.mean(slm_oracle_success)) if slm_oracle_success else 0.0,
                'EM': float(np.mean(em_success)) if em_success else 0.0,
                'ECM': float(np.mean(ecm_success)) if ecm_success else 0.0,
            },
        )


        
    def create_results_summary(self, experiment_results: Dict):
        """
        Generate individual publication-ready figures and tables.
        
        Parameters:
        -----------
        experiment_results : dict
            Complete experiment results
        """
        # Export comprehensive results to Excel (without runtime comparison)
        self._export_full_results_to_excel(experiment_results)
        
        # Generate parameter estimation table (Table 2 style)
        if 'analysis' in experiment_results and 'summary_table' in experiment_results['analysis']:
            self._export_summary_table_to_excel(experiment_results['analysis']['summary_table'])
            
    def _export_mse_to_excel(self, slm_metrics: Dict, bcd_metrics: Dict, em_metrics: Dict, ecm_metrics: Dict):
        """Export MSE comparison to Excel file."""

        params = ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']
        
        data = []
        for param in params:
            key = f'mse_{param}'
            row = {
                'Parameter': param,
                'SLM_mean': slm_metrics.get(key, {}).get('mean', 0) * 100,
                'SLM_std': slm_metrics.get(key, {}).get('std', 0) * 100,
                'BCD_mean': bcd_metrics.get(key, {}).get('mean', 0) * 100,
                'BCD_std': bcd_metrics.get(key, {}).get('std', 0) * 100,
                'EM_mean': em_metrics.get(key, {}).get('mean', 0) * 100,
                'EM_std': em_metrics.get(key, {}).get('std', 0) * 100,
                'ECM_mean': ecm_metrics.get(key, {}).get('mean', 0) * 100,
                'ECM_std': ecm_metrics.get(key, {}).get('std', 0) * 100
            }

            data.append(row)
            
        df = pd.DataFrame(data)
        excel_path = os.path.join(self.figure_dir, 'Table_2_MSE_Comparison.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='MSE_Comparison', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['MSE_Comparison']
            for column in df:
                column_width = max(df[column].astype(str).map(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                worksheet.column_dimensions[chr(65 + col_idx)].width = column_width + 2
                
    def _export_convergence_table_to_excel(
        self,
        slm_iterations: List[int],
        bcd_slm_iterations: List[int],
        slm_joint_iterations: List[int],
        slm_oracle_iterations: List[int],
        em_iterations: List[int],
        ecm_iterations: List[int],
        *,
        success_rates: Optional[Dict[str, float]] = None,
    ):
        """Export convergence comparison table to Excel file."""
        success_rates = success_rates or {}

        def _sr(name: str) -> float:
            v = success_rates.get(name, 0.0)
            try:
                return float(v)
            except Exception:
                return 0.0

        # Calculate statistics for convergence comparison
        convergence_data = {
            'Algorithm': ['SLM-Fixed', 'BCD-SLM', 'SLM-Joint', 'SLM-Oracle', 'EM', 'ECM'],
            'Mean_Iterations': [
                np.mean(slm_iterations) if slm_iterations else 0,
                np.mean(bcd_slm_iterations) if bcd_slm_iterations else 0,
                np.mean(slm_joint_iterations) if slm_joint_iterations else 0,
                np.mean(slm_oracle_iterations) if slm_oracle_iterations else 0,
                np.mean(em_iterations) if em_iterations else 0,
                np.mean(ecm_iterations) if ecm_iterations else 0,
            ],
            'Std_Iterations': [
                np.std(slm_iterations) if slm_iterations else 0,
                np.std(bcd_slm_iterations) if bcd_slm_iterations else 0,
                np.std(slm_joint_iterations) if slm_joint_iterations else 0,
                np.std(slm_oracle_iterations) if slm_oracle_iterations else 0,
                np.std(em_iterations) if em_iterations else 0,
                np.std(ecm_iterations) if ecm_iterations else 0,
            ],
            'Min_Iterations': [
                np.min(slm_iterations) if slm_iterations else 0,
                np.min(bcd_slm_iterations) if bcd_slm_iterations else 0,
                np.min(slm_joint_iterations) if slm_joint_iterations else 0,
                np.min(slm_oracle_iterations) if slm_oracle_iterations else 0,
                np.min(em_iterations) if em_iterations else 0,
                np.min(ecm_iterations) if ecm_iterations else 0,
            ],
            'Max_Iterations': [
                np.max(slm_iterations) if slm_iterations else 0,
                np.max(bcd_slm_iterations) if bcd_slm_iterations else 0,
                np.max(slm_joint_iterations) if slm_joint_iterations else 0,
                np.max(slm_oracle_iterations) if slm_oracle_iterations else 0,
                np.max(em_iterations) if em_iterations else 0,
                np.max(ecm_iterations) if ecm_iterations else 0,
            ],
            'Median_Iterations': [
                np.median(slm_iterations) if slm_iterations else 0,
                np.median(bcd_slm_iterations) if bcd_slm_iterations else 0,
                np.median(slm_joint_iterations) if slm_joint_iterations else 0,
                np.median(slm_oracle_iterations) if slm_oracle_iterations else 0,
                np.median(em_iterations) if em_iterations else 0,
                np.median(ecm_iterations) if ecm_iterations else 0,
            ],
            'Success_Rate': [
                _sr('SLM-Fixed'),
                _sr('BCD-SLM'),
                _sr('SLM-Joint'),
                _sr('SLM-Oracle'),
                _sr('EM'),
                _sr('ECM'),
            ],
        }

        
        df_convergence = pd.DataFrame(convergence_data)
        
        # Format the dataframe for better presentation
        df_convergence['Mean_Iterations'] = df_convergence['Mean_Iterations'].round(1)
        df_convergence['Std_Iterations'] = df_convergence['Std_Iterations'].round(1)
        df_convergence['Median_Iterations'] = df_convergence['Median_Iterations'].round(1)
        
        excel_path = os.path.join(self.figure_dir, 'Table_3_Convergence_Comparison.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_convergence.to_excel(writer, sheet_name='Convergence_Statistics', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Convergence_Statistics']
            for column in df_convergence:
                column_width = max(df_convergence[column].astype(str).map(len).max(), len(column))
                col_idx = df_convergence.columns.get_loc(column)
                worksheet.column_dimensions[chr(65 + col_idx)].width = column_width + 2
            
    def _export_full_results_to_excel(self, experiment_results: Dict):
        """Export comprehensive results to Excel workbook."""
        excel_path = os.path.join(self.figure_dir, 'Full_Experiment_Results.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Experiment configuration
            config_data = []
            if 'config' in experiment_results:
                config = experiment_results['config']
                config_data.append({
                    'Setting': 'Model Dimensions',
                    'Value': f"p={config['model']['p']}, q={config['model']['q']}, r={config['model']['r']}"
                })
                config_data.append({
                    'Setting': 'Sample Size',
                    'Value': config['model']['n_samples']
                })
                config_data.append({
                    'Setting': 'Number of Trials',
                    'Value': config['experiment']['n_trials']
                })
                config_data.append({
                    'Setting': 'Starting Points',
                    'Value': config['algorithms']['common']['n_starts']
                })
                config_data.append({
                    'Setting': 'Random Seed',
                    'Value': config['experiment']['random_seed']
                })
            pd.DataFrame(config_data).to_excel(writer, sheet_name='Configuration', index=False)
            
            # Algorithm performance summary (without runtime details)
            if 'analysis' in experiment_results:
                performance_data = []
                for method in ['slm', 'bcd_slm', 'em', 'ecm']:

                    if method in experiment_results['analysis']:
                        method_analysis = experiment_results['analysis'][method]
                        perf_row = {
                            'Algorithm': method.upper()
                        }
                        # Add MSE values for key parameters
                        for param in ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']:
                            key = f'mse_{param}'
                            if key in method_analysis:
                                perf_row[f'MSE_{param}'] = method_analysis[key].get('mean', 0) * 100
                        performance_data.append(perf_row)
                
                if performance_data:
                    pd.DataFrame(performance_data).to_excel(writer, sheet_name='Performance_Summary', index=False)
                
    def _export_summary_table_to_excel(self, summary_table: pd.DataFrame):
        """Export summary table to Excel."""
        excel_path = os.path.join(self.figure_dir, 'Table_1_Parameter_Estimation_Summary.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_table.to_excel(writer, sheet_name='Parameter_Estimation', index=False)
            
            # Format the worksheet
            worksheet = writer.sheets['Parameter_Estimation']
            for column in summary_table:
                column_width = max(summary_table[column].astype(str).map(len).max(), len(column))
                col_idx = summary_table.columns.get_loc(column)
                worksheet.column_dimensions[chr(65 + col_idx)].width = column_width + 2
                
    def save_all_figures(self):
        """Close all figures and save any remaining plots."""
        plt.close('all')
        

class LoadingPlotter:
    """
    Specialized plotting for loading matrices W and C.
    Creates sine function comparison plots as shown in Figures 2-3 of the paper.
    """
    
    def plot_W_comparison(
        self,
        W_true: np.ndarray,
        W_slm: np.ndarray,
        W_em: np.ndarray,
        W_ecm: np.ndarray,
        component_idx: int = 0,
        *,
        W_bcd: Optional[np.ndarray] = None,
        W_slm_oracle: Optional[np.ndarray] = None,
    ) -> plt.Figure:
        """
        Create publication-ready comparison plot for W loading matrix.
        
        Parameters:
        -----------
        W_true, W_slm, W_em, W_ecm : np.ndarray
            True and estimated W matrices
        component_idx : int
            Which component to plot
            
        Returns:
        --------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Extract component and handle sign alignment
        w_true = W_true[:, component_idx]
        w_slm = W_slm[:, component_idx]
        w_em = W_em[:, component_idx]
        w_ecm = W_ecm[:, component_idx]

        w_bcd = None
        if isinstance(W_bcd, np.ndarray) and W_bcd.size > 0:
            w_bcd = W_bcd[:, component_idx]

        w_oracle = None
        if isinstance(W_slm_oracle, np.ndarray) and W_slm_oracle.size > 0:
            w_oracle = W_slm_oracle[:, component_idx]

        
        # Align signs
        try:
            if np.corrcoef(w_slm, w_true)[0, 1] < 0:
                w_slm = -w_slm
        except Exception:
            pass
        if w_bcd is not None:
            try:
                if np.corrcoef(w_bcd, w_true)[0, 1] < 0:
                    w_bcd = -w_bcd
            except Exception:
                pass
        try:
            if np.corrcoef(w_em, w_true)[0, 1] < 0:
                w_em = -w_em

        except Exception:
            pass
        try:
            if np.corrcoef(w_ecm, w_true)[0, 1] < 0:
                w_ecm = -w_ecm
        except Exception:
            pass
        if w_oracle is not None:
            try:
                if np.corrcoef(w_oracle, w_true)[0, 1] < 0:
                    w_oracle = -w_oracle
            except Exception:
                pass
            
        # Plot with publication-quality styling.
        # IMPORTANT: make lines distinguishable even in grayscale printing.
        # We use (color + linestyle + marker shape + thicker linewidths).
        x = np.arange(len(w_true))

        common = {
            "markevery": 1,  # p/q are small in our paper setting; show marker at each index.
            "markersize": 7,
            "markeredgewidth": 1.2,
            "alpha": 0.98,
        }

        ax.plot(
            x,
            w_true,
            color="black",
            linestyle="-",
            linewidth=3.2,
            marker="o",
            markerfacecolor="black",
            markeredgecolor="black",
            label="Ground Truth",
            zorder=6,
            **common,
        )
        ax.plot(
            x,
            w_slm,
            color="#2E7D32",
            linestyle="--",
            linewidth=2.6,
            marker="s",
            markerfacecolor="#2E7D32",
            markeredgecolor="black",
            label="SLM",
            zorder=5,
            **common,
        )
        if w_bcd is not None:
            ax.plot(
                x,
                w_bcd,
                color="#00897B",
                linestyle="-.",
                linewidth=2.4,
                marker="P",
                markerfacecolor="#00897B",
                markeredgecolor="black",
                label="BCD-SLM",
                zorder=4.5,
                **common,
            )
        if w_oracle is not None:

            ax.plot(
                x,
                w_oracle,
                color="#6A1B9A",
                linestyle="-",
                linewidth=2.4,
                marker="*",
                markersize=10,
                markerfacecolor="#6A1B9A",
                markeredgecolor="black",
                label="SLM-Oracle",
                zorder=4,
                alpha=0.98,
                markevery=1,
                markeredgewidth=1.2,
            )
        ax.plot(
            x,
            w_em,
            color="#EF6C00",
            linestyle=":",
            linewidth=2.4,
            marker="^",
            markerfacecolor="#EF6C00",
            markeredgecolor="black",
            label="EM",
            zorder=3,
            **common,
        )
        ax.plot(
            x,
            w_ecm,
            color="#1565C0",
            linestyle="-.",
            linewidth=2.4,
            marker="D",
            markerfacecolor="#1565C0",
            markeredgecolor="black",
            label="ECM",
            zorder=2,
            **common,
        )


        
        ax.set_xlabel('Variable Index', fontweight='bold')
        ax.set_ylabel('Loading Value', fontweight='bold')
        ax.set_title(f'Loading Matrix W (Component {component_idx+1})', fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set axis limits with some padding
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
        
        plt.tight_layout()
        return fig
        
    def plot_C_comparison(
        self,
        C_true: np.ndarray,
        C_slm: np.ndarray,
        C_em: np.ndarray,
        C_ecm: np.ndarray,
        component_idx: int = 0,
        *,
        C_bcd: Optional[np.ndarray] = None,
        C_slm_oracle: Optional[np.ndarray] = None,
    ) -> plt.Figure:
        """
        Create publication-ready comparison plot for C loading matrix.
        
        Parameters:
        -----------
        C_true, C_slm, C_em, C_ecm : np.ndarray
            True and estimated C matrices
        component_idx : int
            Which component to plot
            
        Returns:
        --------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Extract component and handle sign alignment
        c_true = C_true[:, component_idx]
        c_slm = C_slm[:, component_idx]
        c_em = C_em[:, component_idx]
        c_ecm = C_ecm[:, component_idx]

        c_bcd = None
        if isinstance(C_bcd, np.ndarray) and C_bcd.size > 0:
            c_bcd = C_bcd[:, component_idx]

        c_oracle = None
        if isinstance(C_slm_oracle, np.ndarray) and C_slm_oracle.size > 0:
            c_oracle = C_slm_oracle[:, component_idx]

        
        # Align signs
        try:
            if np.corrcoef(c_slm, c_true)[0, 1] < 0:
                c_slm = -c_slm
        except Exception:
            pass
        if c_bcd is not None:
            try:
                if np.corrcoef(c_bcd, c_true)[0, 1] < 0:
                    c_bcd = -c_bcd
            except Exception:
                pass
        try:
            if np.corrcoef(c_em, c_true)[0, 1] < 0:
                c_em = -c_em

        except Exception:
            pass
        try:
            if np.corrcoef(c_ecm, c_true)[0, 1] < 0:
                c_ecm = -c_ecm
        except Exception:
            pass
        if c_oracle is not None:
            try:
                if np.corrcoef(c_oracle, c_true)[0, 1] < 0:
                    c_oracle = -c_oracle
            except Exception:
                pass
            
        # Plot with publication-quality styling.
        # IMPORTANT: make lines distinguishable even in grayscale printing.
        # We use (color + linestyle + marker shape + thicker linewidths).
        x = np.arange(len(c_true))

        common = {
            "markevery": 1,
            "markersize": 7,
            "markeredgewidth": 1.2,
            "alpha": 0.98,
        }

        ax.plot(
            x,
            c_true,
            color="black",
            linestyle="-",
            linewidth=3.2,
            marker="o",
            markerfacecolor="black",
            markeredgecolor="black",
            label="Ground Truth",
            zorder=6,
            **common,
        )
        ax.plot(
            x,
            c_slm,
            color="#2E7D32",
            linestyle="--",
            linewidth=2.6,
            marker="s",
            markerfacecolor="#2E7D32",
            markeredgecolor="black",
            label="SLM",
            zorder=5,
            **common,
        )
        if c_bcd is not None:
            ax.plot(
                x,
                c_bcd,
                color="#00897B",
                linestyle="-.",
                linewidth=2.4,
                marker="P",
                markerfacecolor="#00897B",
                markeredgecolor="black",
                label="BCD-SLM",
                zorder=4.5,
                **common,
            )
        if c_oracle is not None:

            ax.plot(
                x,
                c_oracle,
                color="#6A1B9A",
                linestyle="-",
                linewidth=2.4,
                marker="*",
                markersize=10,
                markerfacecolor="#6A1B9A",
                markeredgecolor="black",
                label="SLM-Oracle",
                zorder=4,
                alpha=0.98,
                markevery=1,
                markeredgewidth=1.2,
            )
        ax.plot(
            x,
            c_em,
            color="#EF6C00",
            linestyle=":",
            linewidth=2.4,
            marker="^",
            markerfacecolor="#EF6C00",
            markeredgecolor="black",
            label="EM",
            zorder=3,
            **common,
        )
        ax.plot(
            x,
            c_ecm,
            color="#1565C0",
            linestyle="-.",
            linewidth=2.4,
            marker="D",
            markerfacecolor="#1565C0",
            markeredgecolor="black",
            label="ECM",
            zorder=2,
            **common,
        )


        
        ax.set_xlabel('Variable Index', fontweight='bold')
        ax.set_ylabel('Loading Value', fontweight='bold')
        ax.set_title(f'Loading Matrix C (Component {component_idx+1})', fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set axis limits with some padding
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
        
        plt.tight_layout()
        return fig
    
    def plot_component_wise(self, W_list: List[np.ndarray], 
                          C_list: List[np.ndarray], labels: List[str]):
        """
        Plot all components of W and C matrices.
        
        Parameters:
        -----------
        W_list : List[np.ndarray]
            List of W matrices to compare
        C_list : List[np.ndarray]
            List of C matrices to compare
        labels : List[str]
            Labels for each matrix
        """
        r = W_list[0].shape[1]  # Number of components
        
        # Create subplots for W
        fig1, axes1 = plt.subplots(1, r, figsize=(5*r, 5))
        if r == 1:
            axes1 = [axes1]
            
        for j in range(r):
            for i, (W, label) in enumerate(zip(W_list, labels)):
                axes1[j].plot(W[:, j], label=label, linewidth=2)
            axes1[j].set_title(f'W Component {j+1}')
            axes1[j].legend()
            axes1[j].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # Create subplots for C
        fig2, axes2 = plt.subplots(1, r, figsize=(5*r, 5))
        if r == 1:
            axes2 = [axes2]
            
        for j in range(r):
            for i, (C, label) in enumerate(zip(C_list, labels)):
                axes2[j].plot(C[:, j], label=label, linewidth=2)
            axes2[j].set_title(f'C Component {j+1}')
            axes2[j].legend()
            axes2[j].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
    def add_statistical_bands(self, ax: plt.Axes, data: np.ndarray, 
                            color: str = 'blue', alpha: float = 0.2):
        """
        Add confidence/prediction bands to a plot.
        
        Parameters:
        -----------
        ax : plt.Axes
            Axes to add bands to
        data : np.ndarray
            Data array (trials × variables)
        color : str
            Color for the bands
        alpha : float
            Transparency level
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        x = np.arange(len(mean))
        
        # Add ±1 std band
        ax.fill_between(x, mean - std, mean + std, 
                       color=color, alpha=alpha, label='±1 STD')
        
        # Add ±2 std band
        ax.fill_between(x, mean - 2*std, mean + 2*std, 
                       color=color, alpha=alpha/2, label='±2 STD')
        
    def customize_axes(self, ax: plt.Axes, title: str, 
                      xlabel: str, ylabel: str):
        """
        Setup axes labels and formatting.
        
        Parameters:
        -----------
        ax : plt.Axes
            Axes to customize
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        """
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
    def create_loading_heatmap(self, ax: plt.Axes, loading: np.ndarray, 
                             title: str):
        """
        Create heatmap visualization of loading matrix.
        
        Parameters:
        -----------
        ax : plt.Axes
            Axes for the heatmap
        loading : np.ndarray
            Loading matrix (W or C)
        title : str
            Title for the heatmap
        """
        im = ax.imshow(loading.T, aspect='auto', cmap='RdBu_r', 
                      vmin=-np.max(np.abs(loading)), 
                      vmax=np.max(np.abs(loading)))
        ax.set_title(title)
        ax.set_xlabel('Variable Index')
        ax.set_ylabel('Component')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        

class PerformancePlotter:
    """
    Visualization of algorithm performance metrics.
    """
    
    def plot_mse_bars(self, slm_metrics: Dict, bcd_metrics: Dict, em_metrics: Dict, 
                     ecm_metrics: Dict) -> plt.Figure:
        """
        Create publication-ready bar plot comparing MSE across methods.
        
        Parameters:
        -----------
        slm_metrics : dict
            SLM performance metrics
        em_metrics : dict
            EM performance metrics
        ecm_metrics : dict
            ECM performance metrics
            
        Returns:
        --------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        params = ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']
        param_labels = ['$\\mathbf{W}$', '$\\mathbf{C}$', '$\\mathbf{B}$', 
                       '$\\mathbf{\\Sigma}_t$', '$\\sigma_h^2$']
        
        x = np.arange(len(params))
        width = 0.20
        
        # Extract MSE values and multiply by 100 for readability
        slm_mse = [slm_metrics.get(f'mse_{p}', {}).get('mean', 0) * 100 for p in params]
        bcd_mse = [bcd_metrics.get(f'mse_{p}', {}).get('mean', 0) * 100 for p in params]
        em_mse = [em_metrics.get(f'mse_{p}', {}).get('mean', 0) * 100 for p in params]
        ecm_mse = [ecm_metrics.get(f'mse_{p}', {}).get('mean', 0) * 100 for p in params]
        
        # Create bars
        bars1 = ax.bar(x - 1.5 * width, slm_mse, width, label='SLM-Fixed',
                      color='#2E7D32', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x - 0.5 * width, bcd_mse, width, label='BCD-SLM',
                      color='#00897B', alpha=0.8, edgecolor='black', linewidth=1)
        bars3 = ax.bar(x + 0.5 * width, em_mse, width, label='EM',
                      color='#C62828', alpha=0.8, edgecolor='black', linewidth=1)
        bars4 = ax.bar(x + 1.5 * width, ecm_mse, width, label='ECM',
                      color='#1565C0', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:

            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Parameter', fontweight='bold')
        ax.set_ylabel('MSE ($\\times 10^{-2}$)', fontweight='bold')
        ax.set_title('Mean Squared Error Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    def plot_mse_comparison(self, slm_metrics: Dict, em_metrics: Dict, 
                           ecm_metrics: Dict) -> plt.Figure:
        """
        Create box plots comparing MSE across all three methods.
        
        Parameters:
        -----------
        slm_metrics : dict
            SLM performance metrics
        em_metrics : dict
            EM performance metrics
        ecm_metrics : dict
            ECM performance metrics
            
        Returns:
        --------
        fig : plt.Figure
            Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        params = ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']
        param_labels = ['W', 'C', 'B', 'Σₜ', 'σ²ₕ']
        
        for i, (param, label) in enumerate(zip(params, param_labels)):
            ax = axes[i]
            key = f'mse_{param}'
            
            # Prepare data
            data_to_plot = []
            labels = []
            colors = []
            
            if key in slm_metrics:
                if isinstance(slm_metrics[key], dict) and 'values' in slm_metrics[key]:
                    data_to_plot.append(np.array(slm_metrics[key]['values']) * 100)
                else:
                    mean = slm_metrics[key].get('mean', 0) * 100
                    std = slm_metrics[key].get('std', 0) * 100
                    data_to_plot.append(np.random.normal(mean, std, 100))
                labels.append('SLM')
                colors.append('green')
                
            if key in em_metrics:
                if isinstance(em_metrics[key], dict) and 'values' in em_metrics[key]:
                    data_to_plot.append(np.array(em_metrics[key]['values']) * 100)
                else:
                    mean = em_metrics[key].get('mean', 0) * 100
                    std = em_metrics[key].get('std', 0) * 100
                    data_to_plot.append(np.random.normal(mean, std, 100))
                labels.append('EM')
                colors.append('red')
                
            if key in ecm_metrics:
                if isinstance(ecm_metrics[key], dict) and 'values' in ecm_metrics[key]:
                    data_to_plot.append(np.array(ecm_metrics[key]['values']) * 100)
                else:
                    mean = ecm_metrics[key].get('mean', 0) * 100
                    std = ecm_metrics[key].get('std', 0) * 100
                    data_to_plot.append(np.random.normal(mean, std, 100))
                labels.append('ECM')
                colors.append('blue')
                
            # Create box plot
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                    
            ax.set_title(f'MSE for {label}')
            ax.set_ylabel('MSE (×10⁻²)')
            ax.grid(axis='y', alpha=0.3)
            
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        return fig
        
    def plot_bias_variance_tradeoff(self, metrics_dict: Dict) -> plt.Figure:
        """
        Create bias-variance decomposition plots.
        
        Parameters:
        -----------
        metrics_dict : dict
            Dictionary containing bias and variance for each method
            
        Returns:
        --------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        params = ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']
        x = np.arange(len(params))
        width = 0.2
        
        # Extract bias and variance for all three methods
        slm_bias = [metrics_dict.get('slm', {}).get(f'bias_{p}', 0) for p in params]
        slm_var = [metrics_dict.get('slm', {}).get(f'var_{p}', 0) for p in params]
        em_bias = [metrics_dict.get('em', {}).get(f'bias_{p}', 0) for p in params]
        em_var = [metrics_dict.get('em', {}).get(f'var_{p}', 0) for p in params]
        ecm_bias = [metrics_dict.get('ecm', {}).get(f'bias_{p}', 0) for p in params]
        ecm_var = [metrics_dict.get('ecm', {}).get(f'var_{p}', 0) for p in params]
        
        # Create grouped bar plot
        ax.bar(x - width, slm_bias, width/2, label='SLM Bias', 
               color='darkgreen', alpha=0.8)
        ax.bar(x - width, slm_var, width/2, bottom=slm_bias, 
               label='SLM Variance', color='lightgreen', alpha=0.8)
        ax.bar(x, em_bias, width/2, label='EM Bias', 
               color='darkred', alpha=0.8)
        ax.bar(x, em_var, width/2, bottom=em_bias, 
               label='EM Variance', color='lightcoral', alpha=0.8)
        ax.bar(x + width, ecm_bias, width/2, label='ECM Bias', 
               color='darkblue', alpha=0.8)
        ax.bar(x + width, ecm_var, width/2, bottom=ecm_bias, 
               label='ECM Variance', color='lightblue', alpha=0.8)
        
        ax.set_ylabel('Bias² + Variance')
        ax.set_title('Bias-Variance Decomposition')
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        return fig
        
    def plot_computational_time(self, timing_data: Dict) -> plt.Figure:
        """
        Create runtime comparison between all three algorithms.
        
        Parameters:
        -----------
        timing_data : dict
            Timing information for each algorithm
            
        Returns:
        --------
        fig : plt.Figure
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot average times
        methods = list(timing_data.keys())
        avg_times = [timing_data[m].get('mean', 0) for m in methods]
        std_times = [timing_data[m].get('std', 0) for m in methods]
        
        colors = ['green', 'red', 'blue'][:len(methods)]
        bars = ax1.bar(methods, avg_times, yerr=std_times, capsize=10,
                       color=colors, alpha=0.7)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Average Computation Time')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot convergence rates
        conv_rates = [timing_data[m].get('avg_convergence_rate', 0) * 100 for m in methods]
        bars2 = ax2.bar(methods, conv_rates, color=colors, alpha=0.7)
        ax2.set_ylabel('Convergence Rate (%)')
        ax2.set_title('Algorithm Convergence Rates')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 105)
        
        plt.tight_layout()
        return fig
        
    def create_correlation_heatmap(self, correlations: Dict) -> plt.Figure:
        """
        Create heatmap of parameter correlations.
        
        Parameters:
        -----------
        correlations : dict
            Correlation matrices for parameters
            
        Returns:
        --------
        fig : plt.Figure
            Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # SLM correlations
        if 'slm' in correlations:
            sns.heatmap(correlations['slm'], annot=True, fmt='.2f',
                       cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                       ax=axes[0], cbar_kws={'label': 'Correlation'})
            axes[0].set_title('SLM Parameter Correlations')
            
        # EM correlations
        if 'em' in correlations:
            sns.heatmap(correlations['em'], annot=True, fmt='.2f',
                       cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                       ax=axes[1], cbar_kws={'label': 'Correlation'})
            axes[1].set_title('EM Parameter Correlations')
            
        # ECM correlations
        if 'ecm' in correlations:
            sns.heatmap(correlations['ecm'], annot=True, fmt='.2f',
                       cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                       ax=axes[2], cbar_kws={'label': 'Correlation'})
            axes[2].set_title('ECM Parameter Correlations')
            
        plt.tight_layout()
        return fig
        
    def plot_recovery_rates(self, recovery_data: Dict) -> plt.Figure:
        """
        Visualize parameter recovery success rates for all three algorithms.
        
        Parameters:
        -----------
        recovery_data : dict
            Recovery rates for each parameter and method
            
        Returns:
        --------
        fig : plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        params = ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']
        x = np.arange(len(params))
        width = 0.25
        
        # Extract recovery rates
        slm_rates = [recovery_data.get('slm', {}).get(p, 0) * 100 for p in params]
        em_rates = [recovery_data.get('em', {}).get(p, 0) * 100 for p in params]
        ecm_rates = [recovery_data.get('ecm', {}).get(p, 0) * 100 for p in params]
        
        # Create bar plot
        ax.bar(x - width, slm_rates, width, label='SLM', 
               color='green', alpha=0.7)
        ax.bar(x, em_rates, width, label='EM', 
               color='red', alpha=0.7)
        ax.bar(x + width, ecm_rates, width, label='ECM', 
               color='blue', alpha=0.7)
        
        ax.set_ylabel('Recovery Rate (%)')
        ax.set_title('Parameter Recovery Success Rates')
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Add value labels on bars
        for i, (slm, em, ecm) in enumerate(zip(slm_rates, em_rates, ecm_rates)):
            ax.text(i - width, slm + 1, f'{slm:.1f}', 
                   ha='center', va='bottom', fontsize=8)
            ax.text(i, em + 1, f'{em:.1f}', 
                   ha='center', va='bottom', fontsize=8)
            ax.text(i + width, ecm + 1, f'{ecm:.1f}', 
                   ha='center', va='bottom', fontsize=8)
            
        return fig
