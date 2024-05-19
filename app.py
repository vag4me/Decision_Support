import streamlit as st
import pandas as pd
import joblib

# Load the trained model
pipeline = joblib.load('random_forest_model.pkl')

# Custom CSS for styling
st.markdown(
    """
    <style>
       body {
            font-family: Arial, sans-serif;
            background-color: #1e1e1e;
            color: #ffffff;
            margin: 0;
            padding: 0;
        }

        .stTextInput, .stNumberInput {
            margin-bottom: 15px;
            background-color: #333333;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
        }

        .stTextInput input, .stNumberInput input {
            background-color: #444444;
            color: #ffffff;
            border: none;
        }

        .stButton button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 18px;
        }

        .stButton button:hover {
            background-color: #0056b3;
        }

        .stApp header {
            text-align: center;
            margin-top: 20px;
        }

        .form-section {
            background-color: #2c2c2c;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            text-align: center;
        }

        .image1 {
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.1);
        }

        .text-content {
            color: #ffffff;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-size: 16px;
            line-height: 1.5;
            text-align: justify;
            text-justify: inter-word;
            font-weight: bold;
        }

        .prediction {
            background-color: #333333;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.1);
            margin: 20px 0;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)



def main():
    # Display the Tanzania image
    st.image("Tanzania.jpg", caption="Tanzania", use_column_width=True)

    # Display the centered text content
    st.markdown(
        """
        <div style="text-align: center;">
            <p style="color: #ffffff; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 16px; line-height: 1.5; text-align: justify; text-justify: inter-word; font-weight: bold;">
            To calculate your trip's cost in Tanzania, please fill in all required fields on the booking form. This will allow us to provide you with a comprehensive and accurate estimate of your travel expenses.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Input fields
    age_group = st.text_input('Age Group')
    travel_with = st.text_input('Travel With')
    total_female = st.number_input('Total Female', min_value=0, format='%d')
    total_male = st.number_input('Total Male', min_value=0, format='%d')
    purpose = st.text_input('Purpose')
    main_activity = st.text_input('Main Activity')
    info_source = st.text_input('Information Source')
    tour_arrangement = st.text_input('Tour Arrangement')
    package_transport_int = st.text_input('Package Transport International')
    package_accomodation = st.text_input('Package Accommodation')
    package_food = st.text_input('Package Food')
    package_transport_tz = st.text_input('Package Transport Tanzania')
    package_sightseeing = st.text_input('Package Sightseeing')
    package_guided_tour = st.text_input('Package Guided Tour')
    package_insurance = st.text_input('Package Insurance')
    night_mainland = st.number_input('Night Mainland', min_value=0, format='%d')
    night_zanzibar = st.number_input('Night Zanzibar', min_value=0, format='%d')
    first_trip_tz = st.text_input('First Trip to Tanzania')

    # Create a dictionary with the input data
    data = {
        'age_group': [age_group],
        'travel_with': [travel_with],
        'total_female': [total_female],
        'total_male': [total_male],
        'purpose': [purpose],
        'main_activity': [main_activity],
        'info_source': [info_source],
        'tour_arrangement': [tour_arrangement],
        'package_transport_int': [package_transport_int],
        'package_accomodation': [package_accomodation],
        'package_food': [package_food],
        'package_transport_tz': [package_transport_tz],
        'package_sightseeing': [package_sightseeing],
        'package_guided_tour': [package_guided_tour],
        'package_insurance': [package_insurance],
        'night_mainland': [night_mainland],
        'night_zanzibar': [night_zanzibar],
        'first_trip_tz': [first_trip_tz]
    }

    # Convert the data to a DataFrame
    dataframe = pd.DataFrame(data)

    if st.button('Predict'):
        if dataframe.isnull().values.any() or any(dataframe[col].iloc[0] == '' for col in dataframe.columns):
            st.markdown(
                '<div class="prediction"><h4>&#x274C; Error</h4><p>Πρέπει να συμπληρώσεις όλα τα δεδομένα</p></div>',
                unsafe_allow_html=True
            )
            st.image("error.gif", caption="Error", use_column_width=True)
        else:
            y_pred = pipeline.predict(dataframe)
            st.markdown(
                f'<div class="prediction"><h4>&#10003; Prediction</h4><p>Το κόστος θα είναι {y_pred[0]}</p></div>',
                unsafe_allow_html=True
            )
            st.image("giphy.gif", caption="Prediction Result", use_column_width=True)

if __name__ == "__main__":
    main()
