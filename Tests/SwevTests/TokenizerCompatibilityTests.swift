import Testing
@testable import Swev

@Test func roleCapitalizationUsesPortableFilterWithoutChangingOtherExpressions() {
    #expect(TokenizerCompatibility.portableTemplate("{{message['role'].capitalize()}}") == "{{ message['role'] | capitalize }}")
    #expect(TokenizerCompatibility.portableTemplate("{{ item[\"role\"].capitalize() }}") == "{{ item[\"role\"] | capitalize }}")
    let unrelated = "{{ 'capitalize()' }} {{ content.capitalize() }} {{ message['content'] }}"
    #expect(TokenizerCompatibility.portableTemplate(unrelated) == unrelated)
}
