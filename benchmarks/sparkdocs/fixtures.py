"""Publishable synthetic documents and manually labeled benchmark questions."""

from dataclasses import dataclass
from io import BytesIO
from typing import Literal

DocumentFormat = Literal["pdf", "docx", "pptx"]
Split = Literal["development", "holdout"]

_LOCATOR_PREFIX = {"pdf": "PDF page", "docx": "DOCX paragraph", "pptx": "PPTX slide"}


@dataclass(frozen=True, slots=True)
class Evidence:
    locator: str
    text: str


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    text: str
    expected_answer: str | None
    evidence: tuple[Evidence, ...]
    category: str
    answer_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Fixture:
    id: str
    split: Split
    format: DocumentFormat
    passages: tuple[str, ...]
    questions: tuple[Question, ...]

    @property
    def filename(self) -> str:
        return f"{self.id}.{self.format}"

    def render(self) -> bytes:
        if self.format == "pdf":
            import pymupdf

            with pymupdf.open() as document:
                for passage in self.passages:
                    page = document.new_page()
                    space_left = page.insert_textbox(
                        pymupdf.Rect(48, 48, page.rect.width - 48, page.rect.height - 48),
                        passage,
                        fontsize=11,
                    )
                    if space_left < 0:
                        raise ValueError(f"Fixture passage exceeds page bounds: {self.id}")
                return document.tobytes()

        destination = BytesIO()
        if self.format == "docx":
            from docx import Document

            document = Document()
            for passage in self.passages:
                document.add_paragraph(passage)
            document.save(destination)
        else:
            from pptx import Presentation
            from pptx.util import Inches

            presentation = Presentation()
            for passage in self.passages:
                slide = presentation.slides.add_slide(presentation.slide_layouts[6])
                textbox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(5))
                textbox.text_frame.word_wrap = True
                textbox.text_frame.text = passage
            presentation.save(destination)
        return destination.getvalue()


def _fixture(
    fixture_id: str,
    split: Split,
    document_format: DocumentFormat,
    passages: tuple[str, ...],
    labels: tuple[tuple[str, str, str | None, tuple[int, ...], tuple[str, ...]], ...],
) -> Fixture:
    prefix = _LOCATOR_PREFIX[document_format]
    questions = tuple(
        Question(
            id=f"{fixture_id}-{number:02}",
            text=question,
            expected_answer=answer,
            evidence=tuple(
                Evidence(f"{prefix} {index}", passages[index - 1]) for index in positions
            ),
            category=category,
            answer_markers=markers,
        )
        for number, (category, question, answer, positions, markers) in enumerate(labels, start=1)
    )
    return Fixture(fixture_id, split, document_format, passages, questions)


CORPUS = (
    _fixture(
        "cedar-access-policy",
        "development",
        "pdf",
        (
            "Cedar access policy. Scope: corporate office badges and visitor escorts. The policy "
            "owner is Workplace Services. This issue supersedes the February office memo.",
            "Permanent employees must return their physical badge within 10 business days of "
            "their last working day. Remote employees may use the prepaid mailer.",
            "External contractors must return their physical badge within 2 business days of "
            "their last working day. Their sponsor confirms receipt in the access register.",
            "Use form CDR-19B when reporting a badge stolen outside a company site. For a badge "
            "lost in the office, use CDR-19A and contact the lobby desk.",
            "Visitor escorts must remain with their guests in the equipment wing. The escort "
            "requirement ends only when the guest exits the wing.",
            "An overnight visitor permit needs Security Operations approval before the permit can "
            "be issued. Standard visitor passes can be issued by the lobby desk.",
            "Overnight visitor permit requests must reach the approvals queue by 16:00 local "
            "time on the prior business day. Later requests roll to the next available date.",
            "Badge replacement and emergency access are handled by the lobby desk. Temporary "
            "badges are available during desk hours and expire at the end of the day.",
            "Badge types. The office issues permanent photo badges, renewable contractor badges, "
            "day visitor badges, and escorted delivery slips. The colors rotate annually.",
            "Badge activation. Activate a replacement badge at the lobby desk before the first "
            "use. The desk verifies the holder against the workplace directory.",
            "Employee desk moves. Employees who move between floors keep their current physical "
            "badge. Their new team raises a permission change with Workplace Services.",
            "Contractor sponsor changes. If the sponsor changes while work continues, update "
            "the contractor badge record. Keep the original issue date in the access register.",
            "Vendor deliveries. Delivery personnel who remain at the receiving dock are recorded "
            "in the courier register. They do not receive a physical contractor badge.",
            "Contract extensions. A sponsor who extends an external contractor's assignment "
            "submits the updated last working day so the existing badge remains active.",
            "Badge return by mail. Workplace Services records the receipt date once it scans "
            "the badge envelope. A shipping notification alone does not close the record.",
            "Return of temporary badges. Day visitors hand their badges to the lobby desk when "
            "leaving the site. Escorted guests can hand the badge to their escort at the exit.",
            "Quarterly inventory. Compare issued and returned physical badges with the access "
            "register. Investigate records marked returned when the badge is still active.",
            "Badge status on leave. A temporary suspension for extended leave disables door "
            "access while retaining the badge record. It is distinct from ending an assignment.",
            "Review of missed returns. A sponsor checks the access register and contacts the "
            "former holder. The sponsor records any alternate arrangements in the case notes.",
            "Permit records. Store overnight visitor permit approvals in the access register "
            "under the escort's office location. The register records the approval time.",
            "Visitor itinerary changes. An escort updates the planned arrival and departure "
            "times in the permit queue after confirming the changes with their guest.",
        ),
        (
            (
                "exact_identifier",
                "What is form CDR-19B for?",
                "A badge stolen off-site.",
                (4,),
                ("stolen", "badge"),
            ),
            (
                "paraphrase",
                "When can a host leave a guest on their own?",
                "Once the guest exits the equipment wing.",
                (5,),
                ("equipment wing",),
            ),
            (
                "lookalike",
                "How soon must a vendor's worker hand back their badge after their last day?",
                "Within 2 business days.",
                (3,),
                ("2 business days",),
            ),
            (
                "multi_passage",
                "Who approves overnight visitor permits, and what is the request cutoff?",
                "Security Operations; 16:00 local time on the previous business day.",
                (6, 7),
                ("security operations", "16:00", "business day"),
            ),
            ("missing_answer", "What fee applies to replacing a lost Cedar badge?", None, (), ()),
        ),
    ),
    _fixture(
        "harbor-dispatch-runbook",
        "development",
        "docx",
        (
            "Harbor dispatch runbook. The control desk owns weekday routing and after-hours crew "
            "notifications. Read this alongside the monthly route roster.",
            "A standard ambient delivery can wait in staging for 90 minutes after scan-in. Move "
            "unclaimed parcels to the overflow lane at the end of the window.",
            "A refrigerated delivery can wait in staging for 20 minutes after scan-in. Then move "
            "it to cold storage and log the transfer.",
            "Exception HBR-71 is opened when a barcode is visible but cannot be scanned. Add a "
            "photo to the dispatch ticket. HBR-72 is for an absent barcode.",
            "If a courier cannot enter the loading bay, direct them to the west gate and keep the "
            "dock lane clear. The gate attendant issues a temporary access slip.",
            "The route controller authorizes weekend van swaps. Couriers should keep both van "
            "registration numbers handy when requesting the swap.",
            "Record weekend van swaps in the Route Changes ledger once the controller has "
            "approved them. Ordinary weekday changes go into the dispatch ticket.",
            "Cold storage temperatures are checked at the start of each shift. Report unexpected "
            "readings to the shift supervisor before transferring stock.",
        ),
        (
            (
                "exact_identifier",
                "What triggers HBR-71?",
                "A visible barcode that will not scan.",
                (4,),
                ("barcode", "scan"),
            ),
            (
                "paraphrase",
                "Where should drivers go if they can't get into the loading area?",
                "The west gate.",
                (5,),
                ("west gate",),
            ),
            (
                "lookalike",
                "How long can a chilled delivery stay in staging after check-in?",
                "20 minutes.",
                (3,),
                ("20 minutes",),
            ),
            (
                "multi_passage",
                "Who signs off on a weekend van swap, and where does it get recorded?",
                "The route controller; the Route Changes ledger.",
                (6, 7),
                ("route controller", "route changes ledger"),
            ),
            (
                "missing_answer",
                "What temperature in degrees Celsius should cold storage maintain?",
                None,
                (),
                (),
            ),
        ),
    ),
    _fixture(
        "juniper-trial-briefing",
        "development",
        "pptx",
        (
            "Juniper trial briefing\nVolunteer onboarding for a fictional transit trial. "
            "The briefing covers passes, safety incidents, and equipment handover.",
            "Staff passes\nStaff can renew a trial pass until 18:00 on its expiration date. The "
            "desk opens at 08:00 on weekdays.",
            "Volunteer passes\nVolunteers must renew a trial pass by 12:00 on the day before it "
            "expires. Late renewals are treated as new applications.",
            "Incident codes\nUse JNP-42 for a jammed gate that has already been secured. Use "
            "JNP-43 for a gate that remains open.",
            "Lost property\nPut unidentified items from the vehicle in the blue locker. Keep "
            "known owner details with the help desk.",
            "Radio checkout\nThe shift lead approves spare radio loans. A loan is available only "
            "to people already on the deployment roster.",
            "Radio return\nReturn loaned radios to the equipment cabinet by the end of the shift. "
            "Place damaged units in the marked tray.",
            "Trial travel\nParticipants record their journeys in a weekly diary. The trial team "
            "collects diaries on Fridays.",
        ),
        (
            (
                "exact_identifier",
                "When is JNP-42 used?",
                "For a jammed gate that has already been secured.",
                (4,),
                ("jammed", "secured"),
            ),
            (
                "paraphrase",
                "Where do I put belongings found on board when I don't know whose they are?",
                "The blue locker.",
                (5,),
                ("blue locker",),
            ),
            (
                "lookalike",
                "When's the latest someone volunteering can extend their expiring pass?",
                "12:00 the day before it expires.",
                (3,),
                ("12:00", "day before"),
            ),
            (
                "multi_passage",
                "Who approves spare radio loans, and when and where are they returned?",
                "The shift lead; to the equipment cabinet by the end of the shift.",
                (6, 7),
                ("shift lead", "equipment cabinet", "end of the shift"),
            ),
            (
                "missing_answer",
                "How much does the trial reimburse participants per journey?",
                None,
                (),
                (),
            ),
        ),
    ),
    _fixture(
        "maple-claims-handbook",
        "holdout",
        "pdf",
        (
            "Maple claims handbook. This fictional procedure covers claims for equipment "
            "covered under the municipal lending program.",
            "Battery claims for indoor devices must be filed within 30 calendar days of "
            "discovery. The device must be available for assessment.",
            "Battery claims for outdoor devices must be filed within 7 calendar days of "
            "discovery. Leave the device in its existing condition until assessment.",
            "Form MPL-06C covers cracked housings that remain intact. MPL-06D covers complete "
            "separation of the housing. Include two photos with either form.",
            "When the item owner cannot be reached, suspend the claim in the Awaiting Contact "
            "queue. Do not mark it resolved solely because the owner is unresponsive.",
            "Appeals concerning a rejected loan-kit claim are reviewed by the Claims Review "
            "Panel. Initial claim screening remains with the intake coordinator.",
            "Send loan-kit appeals through the Member Portal within 14 calendar days of the "
            "original decision. Attach the decision reference to the appeal.",
            "Replacement kits can be loaned during a pending assessment, subject to inventory. "
            "The equipment desk maintains the availability list.",
        ),
        (
            (
                "exact_identifier",
                "What defect belongs on MPL-06C?",
                "A cracked housing that remains intact.",
                (4,),
                ("cracked", "intact"),
            ),
            (
                "paraphrase",
                "What should I do with the ticket if the equipment owner is unreachable?",
                "Suspend it in the Awaiting Contact queue.",
                (5,),
                ("awaiting contact",),
            ),
            (
                "lookalike",
                "How much time do I have to submit a power-cell claim for a unit used outdoors?",
                "7 calendar days from discovery.",
                (3,),
                ("7 calendar days",),
            ),
            (
                "multi_passage",
                "Who reviews a rejected loan-kit claim appeal, and how and when is it submitted?",
                "The Claims Review Panel; via the Member Portal within 14 calendar days of the "
                "decision.",
                (6, 7),
                ("claims review panel", "member portal", "14 calendar days"),
            ),
            (
                "missing_answer",
                "How many replacement kits can one member borrow at once?",
                None,
                (),
                (),
            ),
        ),
    ),
    _fixture(
        "quartz-returns-guide",
        "holdout",
        "docx",
        (
            "Quartz equipment returns guide. This fictional field program leases devices to "
            "partner offices and to individuals. The customer team manages returns.",
            "Partner offices must send devices back within 21 calendar days of the contract "
            "end date. Keep the shipping receipt on file.",
            "Individual borrowers must send devices back within 5 calendar days of the contract "
            "end date. Keep the shipping receipt until the return is accepted.",
            "Use return code QTZ-R23 for a broken charging cable. Use QTZ-R24 for a missing "
            "charging cable. A broken cable should stay with the device.",
            "If a device powers on but intermittently loses its display, send it to the "
            "diagnostics bench. Do not put it into available stock.",
            "The service desk manager authorizes direct drop-off for battery-damaged devices. "
            "Request authorization before traveling with the device.",
            "Authorized battery-damaged devices go to the rear intake counter. It accepts these "
            "drop-offs between 09:00 and 11:00 on weekdays.",
            "A pickup can be arranged for oversized equipment by opening a pickup request. "
            "Keep the equipment accessible for the assigned driver.",
            "When a declared flood blocks the usual shipping depot, individual borrowers have "
            "until 12 calendar days after the contract end date to send the device back.",
        ),
        (
            (
                "exact_identifier",
                "What does return code QTZ-R23 mean?",
                "A broken charging cable.",
                (4,),
                ("broken", "charging cable"),
            ),
            (
                "paraphrase",
                "Where do units with a screen that cuts out go?",
                "The diagnostics bench.",
                (5,),
                ("diagnostics bench",),
            ),
            (
                "lookalike",
                "How long does a person have to ship their leased device if a declared flood "
                "blocks the regular depot?",
                "12 calendar days from the end of the contract.",
                (9,),
                ("12 calendar days",),
            ),
            (
                "multi_passage",
                "Who authorizes direct drop-off of battery-damaged devices, and where and when "
                "do they go?",
                "The service desk manager; rear intake counter from 09:00 to 11:00 on weekdays.",
                (6, 7),
                ("service desk manager", "rear intake counter", "09:00", "11:00", "weekdays"),
            ),
            (
                "missing_answer",
                "What minimum device weight qualifies for an oversized pickup?",
                None,
                (),
                (),
            ),
        ),
    ),
    _fixture(
        "tidepool-workshop-orientation",
        "holdout",
        "pptx",
        (
            "Tidepool workshop orientation\nThis fictional makerspace runs induction, machine "
            "bookings, and weekend training sessions for enrolled members.",
            "Independent use\nA trained member's laser-cutter reservation can last up to 75 "
            "minutes. They must clear the bed before leaving.",
            "Supervised use\nA new member's supervised laser-cutter reservation can last up to "
            "25 minutes. The supervisor confirms the machine settings before use.",
            "Shutdown reference\nTDP-11 is the checklist for ending work after a ventilation "
            "warning. TDP-12 covers a routine shutdown with no warning.",
            "Consumable storage\nReturn unopened adhesive tubes to the silver bin. Opened tubes "
            "go into the workshop's current-use tray.",
            "Weekend mentors\nThe training coordinator approves switches between weekend "
            "mentor shifts. Both mentors must agree before approval.",
            "Mentor swaps\nEnter approved weekend mentor changes into the Mentor Roster before "
            "17:00 on Thursday. The published schedule is updated that evening.",
            "Membership and bookings\nMembers confirm their attendance through the booking "
            "calendar. Individual cancellations are recorded against the booking.",
        ),
        (
            (
                "exact_identifier",
                "What situation is covered by TDP-11?",
                "Ending work after a ventilation warning.",
                (4,),
                ("ventilation warning",),
            ),
            (
                "paraphrase",
                "Where do sealed tubes of glue belong?",
                "The silver bin.",
                (5,),
                ("silver bin",),
            ),
            (
                "lookalike",
                "What's the longest a newcomer can reserve the laser cutter with a supervisor?",
                "25 minutes.",
                (3,),
                ("25 minutes",),
            ),
            (
                "multi_passage",
                "Who signs off on a weekend mentor swap, and where and by when is it recorded?",
                "The training coordinator; in the Mentor Roster before 17:00 on Thursday.",
                (6, 7),
                ("training coordinator", "mentor roster", "17:00", "thursday"),
            ),
            (
                "missing_answer",
                "What cancellation charge applies if a member misses a workshop?",
                None,
                (),
                (),
            ),
        ),
    ),
)
