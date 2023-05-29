import numpy      as np
import pandas     as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreateCustomersData(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.create_customers_dataframe(X)

        return X

    def create_customers_dataframe(self,  df ):
         """
         To create the customer's data frame is utilized the invoice records associated with each customer.
         """

         data = df.copy()

         # Invoice No
         returned_invoices = data.invoice_no.str.contains(pat='[a-zA-Z]', regex=True)
         canceled = data.query('@returned_invoices').copy()
         purchased = data.query('~@returned_invoices').copy()

         # Quantity
         purchased = purchased.query('quantity > 0')

         # Gross Revenue ( quantity * unit_price )
         purchased['gross_revenue'] = purchased.quantity * purchased.unit_price
         canceled['gross_revenue'] = canceled.quantity * canceled.unit_price

         # ------------------------------------------------------------------------------------------------------------------------
         # Creating the Customers Dataset

         # Customer ID
         customers = pd.concat([purchased.customer_id, canceled.customer_id]).unique()
         customers_ref = pd.DataFrame({'customer_id': customers})

         # Total Purchase
         total_purchase = (purchased.groupby('customer_id')[['gross_revenue']].sum()
                                                                              .reset_index()
                                                                              .rename(columns={'gross_revenue':'total_purchases'}))
         customers_ref = customers_ref.merge(total_purchase, how='left', on='customer_id')

         # Number of products
         number_prod = (purchased.groupby('customer_id')[['stock_code']].count().reset_index()
                                                                        .rename(columns = {'stock_code':'number_products'}))
         customers_ref = customers_ref.merge(number_prod, how='left', on='customer_id')

         # Number of unique produts
         unique_prod = (purchased.groupby('customer_id')[['stock_code']].nunique().reset_index()
                                                                        .rename(columns = {'stock_code':'nunique_products'}))
         customers_ref = customers_ref.merge(unique_prod, how='left', on='customer_id')

         # Average Basket Size
         basket_size = data.groupby(['customer_id', 'invoice_no'])[['quantity']].sum().reset_index()
         avg_basket_size = (basket_size.groupby('customer_id')[['quantity']].mean()
                                                                            .reset_index()
                                                                            .rename(columns = {'quantity':'avg_basket_size'}))

         customers_ref = customers_ref.merge(avg_basket_size, how='left', on='customer_id')

         # Lifetime and Recency
         today = data.invoice_date.max()
         time = purchased.groupby('customer_id')[['invoice_date']].agg({'invoice_date':['min', 'max']}).reset_index()
         time.columns = ['customer_id', 'lifetime', 'recency']
         time.lifetime = (today - time.lifetime).dt.days
         time.recency = (today - time.recency).dt.days

         customers_ref = customers_ref.merge(time, how='left', on='customer_id')

         # Purchase Count
         purchase_count = (data.groupby('customer_id')[['invoice_no']].nunique()
                                                                      .reset_index()
                                                                      .rename(columns={'invoice_no': 'purchase_count'}))

         customers_ref = customers_ref.merge(purchase_count, how='left', on='customer_id')

         # Charge Back Count and Gross Revenue Charge Back
         charge_back = (canceled.groupby('customer_id').agg({'gross_revenue': lambda x: x.sum(),
                                                             'invoice_no': lambda x: x.nunique()})
                                                        .reset_index()
                                                        .rename(columns={'gross_revenue':'gross_revenue_charge_back',
                                                                         'invoice_no':'charge_back_count'}))

         customers_ref = customers_ref.merge(charge_back, how='left', on='customer_id')

         customers_ref[['charge_back_count', 'gross_revenue_charge_back']] = (customers_ref[['charge_back_count', 
                                                                                             'gross_revenue_charge_back']]
                                                                                             .fillna(0))

         # Average Unit Price
         avg_unt_price = (purchased.groupby('customer_id')[['unit_price']].mean()
                                                                          .reset_index()
                                                                          .rename(columns={'unit_price':'avg_unt_price'}))

         customers_ref = customers_ref.merge(avg_unt_price, how='left', on='customer_id')

         # Return Rate
         customers_ref['return_rate'] = customers_ref.charge_back_count / customers_ref.purchase_count

         # Average Purchase Interval
         customers_ref['avg_purchase_interval'] = customers_ref.apply(lambda x: ( x.lifetime / x.purchase_count ) if x.purchase_count > 1 
                                                                                                                  else x.lifetime, axis=1)

         # Frequency
         customers_ref['frequency'] = customers_ref.apply(lambda x: ( x.purchase_count / x.lifetime ) if x.lifetime != 0 
                                                                                                      else 0, axis=1)

         # Gross Revenue
         customers_ref['gross_revenue'] = customers_ref.total_purchases + customers_ref.gross_revenue_charge_back

         # Average Order Value
         customers_ref['avg_order_value'] = customers_ref.gross_revenue / customers_ref.purchase_count

         # Customers Profile
         customers_ref = customers_ref.drop(columns=['total_purchases', 'gross_revenue_charge_back'])

         # Ordenate the columns
         customers_ref = customers_ref[['customer_id', 'lifetime', 'recency', 'avg_purchase_interval', 'frequency', 'number_products', 
                                        'nunique_products', 'avg_basket_size', 'purchase_count', 'charge_back_count', 'return_rate', 
                                        'avg_unt_price', 'avg_order_value', 'gross_revenue']]

         # Positive Gross Revenue
         customers_ref = customers_ref.query('gross_revenue > 1').reset_index(drop=True)

         customers_ref = customers_ref.dropna()
         customers_ref = customers_ref.reset_index(drop=True)

         return customers_ref