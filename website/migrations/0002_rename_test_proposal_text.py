# Generated by Django 3.2.13 on 2022-05-21 08:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='proposal',
            old_name='test',
            new_name='text',
        ),
    ]
