import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class CSVProcessor:
    """AI-powered CSV processor that automatically detects column formats"""
    
    def __init__(self):
        self.df = None
        self.column_mapping = {}
        self.detected_format = None
        
    def load_csv(self, file):
        """Load CSV with intelligent parsing"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise ValueError("Could not decode CSV file")
            
            # Clean column names
            self.df.columns = [col.strip().lower().replace(' ', '_') for col in self.df.columns]
            
            # Auto-detect column types
            self._detect_columns()
            
            return True, "CSV loaded successfully"
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}"
    
    def _detect_columns(self):
        """AI-powered column detection"""
        columns = self.df.columns.tolist()
        
        # Detect date column
        date_keywords = ['date', 'time', 'day', 'transaction']
        for col in columns:
            if any(kw in col for kw in date_keywords):
                self.column_mapping['date'] = col
                # Try to parse dates
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    pass
                break
        
        # Detect product/item column
        product_keywords = ['product', 'item', 'name', 'description', 'sku']
        for col in columns:
            if any(kw in col for kw in product_keywords):
                self.column_mapping['product'] = col
                break
        
        # Detect quantity column
        quantity_keywords = ['quantity', 'qty', 'units', 'count', 'amount']
        for col in columns:
            if any(kw in col for kw in quantity_keywords) and 'price' not in col:
                self.column_mapping['quantity'] = col
                break
        
        # Detect price columns
        price_keywords = ['price', 'cost', 'amount', 'total']
        for col in columns:
            if 'unit' in col or ('price' in col and 'total' not in col):
                self.column_mapping['unit_price'] = col
            elif 'total' in col or ('price' in col and 'total' in col):
                self.column_mapping['total_price'] = col
        
        # If no quantity column found, assume 1
        if 'quantity' not in self.column_mapping:
            self.df['quantity'] = 1
            self.column_mapping['quantity'] = 'quantity'
    
    def get_column_mapping(self):
        """Return detected column mapping"""
        return self.column_mapping
    
    def get_summary_stats(self):
        """Get summary statistics from the data"""
        if self.df is None:
            return None
        
        stats = {
            'total_rows': len(self.df),
            'date_range': None,
            'unique_products': 0,
            'total_revenue': 0,
            'total_transactions': len(self.df)
        }
        
        # Date range
        if 'date' in self.column_mapping:
            date_col = self.column_mapping['date']
            valid_dates = self.df[date_col].dropna()
            if len(valid_dates) > 0:
                stats['date_range'] = {
                    'start': valid_dates.min(),
                    'end': valid_dates.max()
                }
        
        # Unique products
        if 'product' in self.column_mapping:
            stats['unique_products'] = self.df[self.column_mapping['product']].nunique()
        
        # Total revenue
        if 'total_price' in self.column_mapping:
            stats['total_revenue'] = self.df[self.column_mapping['total_price']].sum()
        elif 'unit_price' in self.column_mapping and 'quantity' in self.column_mapping:
            self.df['calculated_total'] = (
                pd.to_numeric(self.df[self.column_mapping['unit_price']], errors='coerce') * 
                pd.to_numeric(self.df[self.column_mapping['quantity']], errors='coerce')
            )
            stats['total_revenue'] = self.df['calculated_total'].sum()
        
        return stats
    
    def analyze_product_performance(self):
        """Analyze individual product performance"""
        if self.df is None or 'product' not in self.column_mapping:
            return []
        
        product_col = self.column_mapping['product']
        quantity_col = self.column_mapping['quantity']
        
        # Group by product
        product_analysis = self.df.groupby(product_col).agg({
            quantity_col: ['sum', 'count', 'mean']
        }).reset_index()
        
        product_analysis.columns = ['product', 'total_quantity', 'transaction_count', 'avg_quantity']
        
        # Calculate weekly velocity (if date is available)
        if 'date' in self.column_mapping:
            date_col = self.column_mapping['date']
            date_range = (self.df[date_col].max() - self.df[date_col].min()).days
            weeks = max(date_range / 7, 1)
            product_analysis['weekly_velocity'] = product_analysis['total_quantity'] / weeks
        
        return product_analysis.to_dict('records')
    
    def detect_trends(self):
        """Detect sales trends and patterns"""
        if self.df is None:
            return {}
        
        trends = {
            'growing_products': [],
            'declining_products': [],
            'seasonal_patterns': []
        }
        
        # This is a simplified trend detection
        # In production, you'd use more sophisticated ML models
        
        if 'date' in self.column_mapping and 'product' in self.column_mapping:
            date_col = self.column_mapping['date']
            product_col = self.column_mapping['product']
            quantity_col = self.column_mapping['quantity']
            
            # Get products and their trends
            for product in self.df[product_col].unique():
                if pd.isna(product):
                    continue
                    
                product_data = self.df[self.df[product_col] == product].copy()
                product_data = product_data.sort_values(date_col)
                
                if len(product_data) >= 4:  # Need at least 4 data points
                    # Simple trend: compare first half vs second half
                    mid_point = len(product_data) // 2
                    first_half = product_data.iloc[:mid_point][quantity_col].sum()
                    second_half = product_data.iloc[mid_point:][quantity_col].sum()
                    
                    if second_half > first_half * 1.2:  # 20% increase
                        trends['growing_products'].append({
                            'product': product,
                            'growth_rate': ((second_half - first_half) / first_half * 100)
                        })
                    elif second_half < first_half * 0.8:  # 20% decrease
                        trends['declining_products'].append({
                            'product': product,
                            'decline_rate': ((first_half - second_half) / first_half * 100)
                        })
        
        return trends
    
    def generate_recommendations(self, inventory=None):
        """Generate AI-powered recommendations based on analysis"""
        recommendations = []
        
        if self.df is None:
            return recommendations
        
        # Analyze product performance
        products = self.analyze_product_performance()
        trends = self.detect_trends()
        
        # Generate reorder recommendations
        for product_data in products:
            product_name = product_data['product']
            
            if pd.isna(product_name):
                continue
            
            # Check if we have inventory data
            current_stock = 0
            if inventory and product_name in inventory:
                current_stock = inventory[product_name].get('quantity', 0)
            
            # Calculate recommendation
            weekly_velocity = product_data.get('weekly_velocity', product_data.get('total_quantity', 0) / 4)
            
            # If stock is low (less than 1 week supply)
            if current_stock < weekly_velocity:
                # Check if product is growing
                is_growing = any(p['product'] == product_name for p in trends.get('growing_products', []))
                
                growth_rate = 0
                if is_growing:
                    growth_item = next(p for p in trends['growing_products'] if p['product'] == product_name)
                    growth_rate = growth_item['growth_rate']
                
                # Calculate order quantity (2 weeks supply + growth buffer)
                order_qty = int(weekly_velocity * 2 * (1 + growth_rate/100))
                
                confidence = 85 + min(growth_rate / 2, 10)  # Higher confidence for growing products
                
                recommendations.append({
                    'type': 'REORDER',
                    'product': product_name,
                    'current_stock': current_stock,
                    'weekly_velocity': round(weekly_velocity, 1),
                    'recommended_quantity': order_qty,
                    'reason': f"Sales velocity: {round(weekly_velocity, 1)} units/week. Current stock: {current_stock} units.",
                    'confidence': round(confidence, 0),
                    'growth_rate': round(growth_rate, 1) if is_growing else 0
                })
        
        return recommendations
    
    def get_dataframe(self):
        """Return the processed dataframe"""
        return self.df