import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


def sendemail(smtp_server: str,
              smtp_port: int,
              email_username: str,
              email_password: str,
              email_subject: str,
              from_address: str,
              to_address_list: list,
              cc_address_list: list,
              message_html: str,
              csv_filename: str,
              image_filename: str):

    # Create message
    message = MIMEMultipart("alternative")
    message["Subject"] = email_subject
    message["From"] = from_address
    message["To"] = ", ".join(to_address_list)
    message["CC"] = ", ".join(cc_address_list)
    message.attach(MIMEText(message_html, "html"))

    # Attach image and csv
    for filename in [csv_filename, image_filename]:
        with open(filename, 'rb') as file:
            # Attach the file with filename to the email
            message.attach(MIMEApplication(file.read(), Name=filename))

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls(context=context)
        server.login(email_username, email_password)
        server.sendmail(from_address,
                        to_address_list + cc_address_list,
                        message.as_string())
