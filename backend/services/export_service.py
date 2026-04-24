import csv
import io
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def export_orders_csv(orders):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Phone", "Status", "Items", "Total", "Date"])
    for o in orders:
        od = o.get("order_data", {})
        items_str = "; ".join(
            f"{i.get('item','')} x{i.get('quantity',1)}"
            for i in od.get("menu_items_ordered", [])
        )
        writer.writerow([
            o["id"], o["phone_number"], o.get("status", "pending"),
            items_str, od.get("total_price", "0"), o["orderdate"],
        ])
    return output.getvalue()


def export_orders_pdf(orders):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Orders Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    data = [["ID", "Phone", "Status", "Total", "Date"]]
    for o in orders:
        od = o.get("order_data", {})
        data.append([
            str(o["id"]), o["phone_number"], o.get("status", "pending"),
            od.get("total_price", "0"), str(o["orderdate"])[:19],
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()
