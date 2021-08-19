import os
import smtplib
import ssl
from typing import List
from email.message import EmailMessage
import mimetypes


def sendemail(smtp_server: str,
              smtp_port: int,
              email_username: str,
              email_password: str,
              email_subject: str,
              from_address: str,
              to_address_list: list,
              cc_address_list: list,
              message_html: str,
              filename_list: List[str]):

    # Create message
    message = EmailMessage()
    message["Subject"] = email_subject
    message["From"] = from_address
    message["To"] = ", ".join(to_address_list)
    message["CC"] = ", ".join(cc_address_list)
    message.add_alternative(message_html, subtype="html")

    # Attach files
    for filename in filename_list:
        ctype, encoding = mimetypes.guess_type(filename)
        maintype, subtype = ctype.split('/', 1)
        with open(filename, 'rb') as file:
            message.add_attachment(file.read(),
                                   maintype=maintype,
                                   subtype=subtype,
                                   filename=os.path.basename(filename))

    # Open SSL connection and send the email
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls(context=context)
        server.login(email_username, email_password)
        server.sendmail(from_address,
                        to_address_list + cc_address_list,
                        message.as_string())
